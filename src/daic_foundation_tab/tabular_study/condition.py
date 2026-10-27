from __future__ import annotations

import time
import tomllib
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab.config import write_config
from daic_foundation_tab.data.labels import load_test_ground_truth
from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.data.validation import _offending_columns, fit_feature_selection
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.predictions import classification_predictions
from daic_foundation_tab.evaluation.regression import regression_metrics, regression_predictions
from daic_foundation_tab.models.registry import create_model
from daic_foundation_tab.runner import _git_commit
from daic_foundation_tab.tabular_study.bundle import file_hash, read_bundle
from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.environment import environment_metadata
from daic_foundation_tab.tracking.lifecycle import RunRecord
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    require_cuda_device,
    reset_cuda_peak_memory,
)
from daic_foundation_tab.tracking.seeds import set_global_seed
from daic_foundation_tab.tracking.wandb import WandbTracker


def _sync(device: str) -> None:
    import torch

    torch.cuda.synchronize(device)


def _dependency_versions() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "tabpfn", "structured-data-models"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _check_dependencies(versions: dict) -> None:
    root = Path(__file__).resolve().parents[3]
    lock = tomllib.loads((root / "uv.lock").read_text())
    expected = {item["name"]: item["version"] for item in lock["package"] if item["name"] in versions}
    if expected != versions:
        raise ValueError("Installed study dependencies do not match uv.lock")


def _source_checkpoint_hash(condition: dict, detail: dict) -> str:
    if condition["mode"] == "frozen_icl":
        digest = detail["checkpoint_sha256"]
    elif condition["family"] == "kumo_medium":
        digest = detail["source_checkpoint_sha256"]
    else:
        from tabpfn.model_loading import resolve_model_path

        which = "classifier" if condition["task"] == "classification" else "regressor"
        paths, _, _, _ = resolve_model_path(None, which=which, version="v3.5")
        if len(paths) != 1 or not paths[0].is_file():
            raise ValueError("TabPFN source checkpoint is unavailable after training")
        digest = file_hash(paths[0])
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Source checkpoint hash is invalid")
    return digest


def _timed(device, function, *args, **kwargs):
    _sync(device)
    started = time.perf_counter()
    value = function(*args, **kwargs)
    _sync(device)
    return value, time.perf_counter() - started


def _condition_config(study: dict, condition: dict, host: dict, attempt_id: str) -> dict:
    config = deepcopy(study["configuration"])
    settings = config.pop("tabular_study")
    profiles = config.pop("profiles")
    model = deepcopy(profiles[f"{condition['family']}:{condition['task']}:{condition['mode']}"])
    model["parameters"]["random_state"] = condition["model_seed"]
    config["model"] = model
    config["project"].update(
        seed=condition["model_seed"],
        output_root=host["output_root"],
        cache_root=host.get("cache_root", host["output_root"]),
    )
    config["runtime"].update(device=host.get("device", "cuda:0"), require_cuda=True)
    config["experiment"].update(
        name=condition["condition_id"][:16],
        task=condition["task"],
        mode=condition["mode"],
        feature_set=condition["modality"],
    )
    config["evaluation"].update(
        threshold=0.5 if condition["task"] == "classification" else None,
        test_predictions=condition["stage"] == "final",
    )
    config["evaluation"]["repeated_holdout"]["enabled"] = False
    config["bootstrap"]["enabled"] = False
    config["study"] = {
        "campaign_id": study["study_id"],
        "condition_id": condition["condition_id"],
        "stage": condition["stage"],
        "group_id": f"{condition['family']}-{condition['modality']}",
        "fold": condition["fold"],
        "model_seed": condition["model_seed"],
        "fold_seed": settings["fold_seed"],
        "attempt_id": attempt_id,
        "host_label": host.get("host_label", "unlabeled"),
        "software_identity": study["software_identity"],
    }
    tracking = config["tracking"]["wandb"]
    tracking.update(group=study["study_id"])
    for key in ("mode", "entity"):
        if key in host.get("wandb", {}):
            tracking[key] = host["wandb"][key]
    return config


def _record_validation(table, selected, roles, condition):
    stats = {}
    for role, ids in roles.items():
        subset = table.loc[ids, selected.columns]
        stats[role] = {
            "participants": len(ids),
            "feature_count": len(selected.columns),
            "missing_cell_percentage": float(subset.isna().to_numpy().mean() * 100),
            "complete_cases": int(subset.notna().all(axis=1).sum()),
        }
    return {
        "feature_set": condition["modality"],
        "feature_count_before_filtering": len(selected.columns)
        + len(selected.dropped_all_nan)
        + len(selected.dropped_constant),
        "dropped_all_nan": selected.dropped_all_nan,
        "dropped_constant": selected.dropped_constant,
        "near_constant": selected.near_constant,
        "reported_splits": list(roles),
        "split_statistics": stats,
        "exclusions": {},
    }


def run_condition(bundle: str | Path, condition: dict, host: dict, attempt_id: str) -> Path:
    study, table, features, folds = read_bundle(bundle)
    if condition not in study["conditions"][condition["stage"]]:
        raise ValueError("Condition is not part of the prepared schedule")
    config = _condition_config(study, condition, host, attempt_id)
    artifacts = RunArtifacts(
        Path(host["output_root"]) / condition["condition_id"],
        config["model"]["name"],
        condition["modality"],
        condition["model_seed"],
    )
    record = RunRecord(artifacts, config)
    logger = configure_run_logger(artifacts.path / "run.log", "INFO")
    tracker = None
    try:
        with record.lifecycle():
            started = time.perf_counter()
            device = require_cuda_device(config["runtime"]["device"])
            set_global_seed(condition["model_seed"])
            reset_cuda_peak_memory(device)
            environment = environment_metadata(device)
            environment.update(git_commit=_git_commit(), host_label=config["study"]["host_label"],
                               dependency_versions=_dependency_versions())
            _check_dependencies(environment["dependency_versions"])
            artifacts.json("environment.json", environment)
            table["participant_id"] = table["participant_id"].astype(str)
            if table["participant_id"].duplicated().any():
                raise ValueError("Bundle contains duplicate participants")
            table = table.set_index("participant_id", drop=False)
            assignments = folds[condition["task"]]
            partition = (
                assignments["final"]
                if condition["stage"] == "final"
                else next(
                    item for item in assignments["folds"] if item["fold"] == condition["fold"]
                )
            )
            roles = {role: partition[role] for role in ("fit", "selection")}
            if condition["stage"] == "cv":
                roles["evaluation"] = partition["evaluation"]
            else:
                roles.update(
                    {
                        split: sorted(table.loc[table["split"] == split, "participant_id"].tolist())
                        for split in ("dev", "test")
                    }
                )
            if any(not ids or not set(ids) <= set(table.index) for ids in roles.values()):
                raise ValueError("Fold participants are missing or empty")
            if sum(map(len, roles.values())) != len(
                set().union(*(set(ids) for ids in roles.values()))
            ):
                raise ValueError("Participants overlap across fold roles")
            if any(
                set(table.loc[roles[role], "split"]) != {"train"} for role in ("fit", "selection")
            ):
                raise ValueError(
                    "Fit and selection participants must belong to the official training split"
                )
            if condition["stage"] == "cv" and set(table.loc[roles["evaluation"], "split"]) != {
                "train"
            }:
                raise ValueError(
                    "CV evaluation participants must belong to the official training split"
                )
            modalities = {
                "audio": {"audio"},
                "visual": {"visual"},
                "audio_visual": {"audio", "visual"},
            }
            columns = features.loc[
                features["modality"].isin(modalities[condition["modality"]]), "column"
            ].tolist()
            if not columns or len(set(columns)) != len(columns) or _offending_columns(columns):
                raise ValueError("Invalid or leaking feature columns")
            selection = fit_feature_selection(table.loc[roles["fit"], columns])
            if not selection.columns:
                raise ValueError("No usable features remain in the fit partition")
            target_column = (
                "target_binary" if condition["task"] == "classification" else "target_phq8"
            )
            dtype = int if condition["task"] == "classification" else float
            fit_x = table.loc[roles["fit"], selection.columns]
            fit_y = table.loc[roles["fit"], target_column].astype(dtype)
            config["study"].update(
                selection_seed=partition["selection_seed"],
                manifest_id=fingerprint(partition),
                cohort_id=fingerprint(sorted(table.index.tolist())),
            )
            record.update(**config["study"], paired_comparison_verified=True,
                          selection_labels_used=condition["mode"] == "fine_tuning")
            artifacts.json(
                "fold_assignments.json",
                {"roles": roles, "manifest_id": config["study"]["manifest_id"]},
            )
            artifacts.json(
                "feature_selection.json",
                {
                    "columns": selection.columns,
                    "dropped_all_nan": selection.dropped_all_nan,
                    "dropped_constant": selection.dropped_constant,
                },
            )
            write_config(config, artifacts.path / "config_resolved.yaml")
            tracker = WandbTracker.start(
                config,
                artifacts,
                _record_validation(table, selection, roles, condition),
                study["study_id"],
            )
            model = create_model(config["model"])
            fit_seconds = context_seconds = 0.0
            selected_checkpoint = None
            if condition["mode"] == "fine_tuning":
                selection_x = table.loc[roles["selection"], selection.columns]
                selection_y = table.loc[roles["selection"], target_column].astype(dtype)
                with phase(logger, "training") as phase_started:

                    def epoch_callback(metrics):
                        tracker.record_fine_tuning_epoch(metrics)
                        epoch = metrics.get("train/epoch")
                        if isinstance(epoch, (int, float)):
                            log_progress(
                                logger,
                                "training",
                                min(int(epoch) + 1, config["model"]["parameters"]["epochs"]),
                                config["model"]["parameters"]["epochs"],
                                phase_started,
                            )

                    _, fit_seconds = _timed(
                        device,
                        model.fit,
                        fit_x,
                        fit_y,
                        validation_features=selection_x,
                        validation_target=selection_y,
                        checkpoint_directory=artifacts.path / "checkpoints",
                        epoch_callback=epoch_callback,
                    )
                model_detail = model.finetune_metadata()
                selected_checkpoint = Path(model_detail["checkpoint_path"]).resolve()
                checkpoint_root = (artifacts.path / "checkpoints").resolve()
                if (
                    not selected_checkpoint.is_relative_to(checkpoint_root)
                    or not selected_checkpoint.is_file()
                    or file_hash(selected_checkpoint) != model_detail["checkpoint_sha256"]
                ):
                    raise ValueError("Selected fine-tuning checkpoint is invalid")
                tracker.record_fine_tuning(
                    fit_y, selection_y, len(selection.columns), fit_seconds, model_detail
                )
            else:
                with phase(logger, "preparing ICL context"):
                    _, context_seconds = _timed(device, model.fit_context, fit_x, fit_y)
                model_detail = model.icl_metadata()
                tracker._log(
                    "icl_context",
                    {
                        "context_rows": len(fit_y),
                        "reserved_validation_rows": len(roles["selection"]),
                        "feature_count": len(selection.columns),
                        "context_setup_seconds": context_seconds,
                    },
                )
            source_hash = _source_checkpoint_hash(condition, model_detail)
            model_detail["source_checkpoint_sha256"] = source_hash
            record.update(source_checkpoint_sha256=source_hash)
            tracker.results["model_provenance"] = {"source_checkpoint_sha256": source_hash}
            tracker._log("model_provenance", {"source_checkpoint_sha256": source_hash})
            artifacts.json("adaptation_metadata.json", model_detail)
            predict_seconds = 0.0
            for role in ["evaluation"] if condition["stage"] == "cv" else ["dev", "test"]:
                frame = table.loc[roles[role], selection.columns]
                with phase(logger, f"predicting {role}"):
                    if condition["task"] == "classification":
                        values, elapsed = _timed(device, model.predict_proba, frame)
                        probabilities = np.asarray(values, dtype=float)
                        if (
                            probabilities.shape != (len(frame), 2)
                            or not np.isfinite(probabilities).all()
                            or not np.all((probabilities >= 0) & (probabilities <= 1))
                            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                        ):
                            raise ValueError("Invalid classification probabilities")
                        values = probabilities[:, 1]
                    else:
                        values, elapsed = _timed(device, model.predict, frame)
                        values = np.asarray(values, dtype=float).reshape(-1)
                        if len(values) != len(frame) or not np.isfinite(values).all():
                            raise ValueError("Invalid regression predictions")
                    predict_seconds += elapsed
                target = None
                if role != "test":
                    target = (
                        table.loc[roles[role], target_column].astype(dtype).reset_index(drop=True)
                    )
                elif host.get("test_ground_truth"):
                    target = load_test_ground_truth(
                        Path(host["test_ground_truth"]),
                        pd.Series(roles[role]),
                        target="binary_depression"
                        if condition["task"] == "classification"
                        else "phq8",
                    )
                ids = pd.Series(roles[role], dtype=str)
                split = "cv" if role == "evaluation" else role
                if condition["task"] == "classification":
                    labels = (values > 0.5).astype(int)
                    predictions = classification_predictions(ids, split, labels, values, target)
                    metrics = (
                        classification_metrics(target, labels, values)
                        if target is not None
                        else None
                    )
                else:
                    predictions = regression_predictions(ids, split, values, target)
                    metrics = regression_metrics(target, values) if target is not None else None
                artifacts.csv(f"predictions_{split}.csv", predictions)
                if metrics is not None:
                    artifacts.json(f"metrics_{split}.json", metrics)
                    if split == "cv":
                        tracker.record_cv(metrics)
                    elif split == "dev":
                        tracker.record_development(metrics)
                    else:
                        tracker.record_test(metrics)
            metadata = model.run_metadata()
            runtime = {
                "fit_seconds": fit_seconds,
                "context_setup_seconds": context_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": time.perf_counter() - started,
                "device": device,
                **cuda_peak_memory(device),
            }
            artifacts.json("runtime.json", runtime)
            artifacts.json("model.json", metadata)
            tracker.complete(metadata, environment, runtime)
            if selected_checkpoint:
                for path in (artifacts.path / "checkpoints").rglob("*"):
                    if (
                        path.is_file()
                        and path.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}
                        and path.resolve() != selected_checkpoint
                    ):
                        path.unlink()
    except BaseException:
        if tracker:
            tracker.fail()
        raise
    return artifacts.path
