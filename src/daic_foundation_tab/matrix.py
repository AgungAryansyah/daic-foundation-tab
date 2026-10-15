from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from daic_foundation_tab.config import load_config
from daic_foundation_tab.tracking.logging import configure_run_logger, phase
from daic_foundation_tab.tracking.runtime import require_cuda_device
from daic_foundation_tab.tracking.test_evaluations import (
    collect_regression_test_evaluations,
    collect_test_evaluations,
)


def _settings(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def verify_icl_run(output: Path, config: dict) -> None:
    def saved(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    if yaml.safe_load((output / "config_resolved.yaml").read_text()) != _settings(config):
        raise ValueError("Saved configuration differs from the current frozen ICL preset")
    regression = config["experiment"]["task"] == "regression"
    summary = saved("dataset_summary.json")
    metadata = saved("icl_metadata.json")
    model = saved("model.json")
    checkpoint = Path(model["checkpoint_path"]).resolve()
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        not model["frozen"]
        or digest != model["checkpoint_sha256"]
        or metadata["checkpoint_sha256"] != digest
        or metadata["weights_frozen"] is not True
        or metadata["gradients_enabled"] is not False
        or metadata["optimizer_steps"] != 0
        or metadata["epochs_completed"] != 0
        or metadata["context_rows"] < 1
        or len(metadata["weights_before_inference_sha256"]) != 64
        or metadata["weights_after_inference_sha256"]
        != metadata["weights_before_inference_sha256"]
    ):
        raise ValueError("Frozen ICL metadata or pretrained checkpoint verification failed")
    family = config["model"]["name"].removesuffix("_regressor").removesuffix("_icl")
    expected_model, expected_package = {
        "tabiclv2": ("TabICLv2-ICL", "tabicl"),
        "tabpfn35": ("TabPFN-3.5-ICL", "tabpfn"),
        "kumo_medium": ("Kumo-Tabular-Medium-ICL", "structured-data-models"),
    }[family]
    if (
        model["model_name"] != expected_model + (" Regressor" if regression else "")
        or model["package"] != expected_package
        or model["checkpoint"] != config["model"]["checkpoint_version"]
        or model["parameters"] != config["model"]["parameters"]
    ):
        raise ValueError("Frozen ICL model identity differs from the configured preset")
    if family == "tabpfn35" and (
        model.get("model_version") != "v3.5" or model.get("package_version") != "9.1.0"
    ):
        raise ValueError("Unexpected TabPFN-3.5 model or package version")
    if family == "tabiclv2" and model.get("package_version") != "2.2.0":
        raise ValueError("Unexpected TabICLv2 package version")
    if family == "kumo_medium" and (
        model.get("package_version") != "0.2.0rc1"
        or model.get("model_version") != "v1.0.1"
        or model.get("package_revision") != "842c408fe2a8711bdf2e7cff4bfbe54266d6b940"
        or model.get("checkpoint_revision") != "3c3e10bbdb590ace29e7026847f92db3c603096d"
        or model.get("size") != "medium"
    ):
        raise ValueError("Unexpected Kumo Medium model or package revision")
    assignment = pd.read_csv(output / "icl_split.csv")
    reserved_rows = math.ceil(
        len(assignment) * config["evaluation"]["fine_tuning"]["validation_fraction"]
    )
    context_rows = len(assignment) - reserved_rows
    if (
        len(assignment) != summary["train"]["participants"]
        or assignment["participant_id"].duplicated().any()
        or set(assignment["role"]) != {"icl_context", "reserved_validation"}
        or metadata["context_rows"] != context_rows
        or int((assignment["role"] == "reserved_validation").sum()) != reserved_rows
    ):
        raise ValueError("Frozen ICL context does not match the saved partition")
    if not regression and any(
        set(assignment.loc[assignment["role"] == role, "target"].astype(int)) != {0, 1}
        for role in ("icl_context", "reserved_validation")
    ):
        raise ValueError("Frozen ICL classification partition lost a target class")
    runtime = saved("runtime.json")
    if runtime["device"] != "cuda:0" or runtime["optimizer_steps"] != 0:
        raise ValueError("Frozen ICL run did not remain inference-only on cuda:0")
    tracking = config["tracking"]["wandb"]
    expected_status = (
        "finished" if tracking["enabled"] and tracking["mode"] != "disabled" else "disabled"
    )
    if saved("wandb_run.json")["status"] != expected_status:
        raise ValueError("Tracking did not finish")
    column = "phq8_pred" if regression else "prob_depressed"
    metric = "mae" if regression else "roc_auc"
    for split in ("dev", "test"):
        predictions = pd.read_csv(output / f"predictions_{split}.csv")
        if (
            len(predictions) != summary[split]["participants"]
            or predictions["participant_id"].duplicated().any()
            or not np.isfinite(predictions[column].to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"Incomplete or invalid {split} predictions")
        if not regression and (
            not np.allclose(predictions["prob_non_depressed"] + predictions["prob_depressed"], 1)
            or not set(predictions["y_pred"].unique()).issubset({0, 1})
        ):
            raise ValueError(f"Invalid {split} classification probabilities")
        metrics = saved(f"metrics_{split}.json")
        if not np.isfinite(metrics[metric]):
            raise ValueError(f"Invalid {split} metrics")
        pd.read_csv(output / f"metrics_{split}.csv")
    if config["bootstrap"]["enabled"]:
        if len(pd.read_csv(output / "bootstrap_dev.csv")) != config["bootstrap"]["iterations"]:
            raise ValueError("Incomplete bootstrap distribution")
        saved("bootstrap_dev_summary.json")
    repeated = config["evaluation"]["repeated_holdout"]
    if repeated["enabled"]:
        seeds = list(range(repeated["seed_start"], repeated["seed_start"] + repeated["repeats"]))
        if pd.read_csv(output / "repeated_holdout_metrics.csv")["seed"].tolist() != seeds:
            raise ValueError("Incomplete repeated holdouts")
        for seed in seeds:
            assignment = pd.read_csv(output / "repeated_splits" / f"seed_{seed:03d}.csv")
            evaluation_rows = math.ceil(
                len(assignment) * repeated["validation_fraction"]
            )
            reserved_rows = math.ceil(
                (len(assignment) - evaluation_rows)
                * config["evaluation"]["fine_tuning"]["validation_fraction"]
            )
            if (
                len(assignment) != summary["train"]["participants"]
                or assignment["participant_id"].duplicated().any()
                or set(assignment["role"]) != {"icl_context", "reserved_validation", "evaluation"}
                or assignment["role"].value_counts().to_dict()
                != {
                    "evaluation": evaluation_rows,
                    "reserved_validation": reserved_rows,
                    "icl_context": len(assignment) - evaluation_rows - reserved_rows,
                }
            ):
                raise ValueError("Invalid repeated frozen ICL partition provenance")
        saved("repeated_holdout_summary.json")
    saved("environment.json")
    (output / "summary.md").read_text()


def verify_run(output: Path, config: dict) -> None:
    if config["experiment"].get("mode", "fine_tuning") == "frozen_icl":
        verify_icl_run(output, config)
        return
    def saved(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    if yaml.safe_load((output / "config_resolved.yaml").read_text()) != _settings(config):
        raise ValueError("Saved configuration differs from the current preset")
    metadata = saved("finetune_metadata.json")
    is_kumo = config["model"]["name"].startswith("kumo_medium")
    checkpoint = Path(metadata["checkpoint_path"]).resolve()
    suffix = ".pt" if is_kumo else ".pth"
    if not checkpoint.is_relative_to(output.resolve()) or checkpoint.suffix != suffix:
        raise ValueError(f"Selected checkpoint must be a {suffix} file inside the run")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash does not match")
    metric = config["model"]["parameters"]["eval_metric"]
    if metadata["selection_metric"] != metric or not math.isfinite(
        metadata["best_validation_metric"]
    ):
        raise ValueError("Selected validation metric is invalid")
    model = saved("model.json")
    if is_kumo:
        if (
            model["package"] != "structured-data-models"
            or model["package_version"] != "0.2.0rc1"
            or model["size"] != "medium"
            or any(model[key] != config["model"][key] for key in ("model_version", "package_revision", "checkpoint_revision"))
            or model["checkpoint"] != config["model"]["checkpoint_version"]
        ):
            raise ValueError("Unexpected Kumo model or package revision")
        if not 0 <= metadata["selected_epoch"] <= metadata["epochs_completed"] <= config["model"]["parameters"]["epochs"] or metadata["optimizer_steps"] < 1:
            raise ValueError("Kumo fine-tuning did not complete valid optimizer steps")
        if not math.isfinite(metadata["baseline_validation_metric"]):
            raise ValueError("Kumo baseline validation metric is invalid")
        if len(metadata["source_checkpoint_sha256"]) != 64:
            raise ValueError("Kumo source checkpoint hash is invalid")
    elif model["model_version"] != "v3.5" or model["package_version"] != "9.1.0":
        raise ValueError("Unexpected TabPFN model or package version")
    if saved("runtime.json")["device"] != "cuda:0":
        raise ValueError("Run did not use cuda:0")
    tracking = config["tracking"]["wandb"]
    expected_status = (
        "finished" if tracking["enabled"] and tracking["mode"] != "disabled" else "disabled"
    )
    if saved("wandb_run.json")["status"] != expected_status:
        raise ValueError("Tracking did not finish")
    summary = saved("dataset_summary.json")
    regression = config["experiment"]["task"] == "regression"
    for split in ("dev", "test"):
        predictions = pd.read_csv(output / f"predictions_{split}.csv")
        if (
            len(predictions) != summary[split]["participants"]
            or predictions["participant_id"].duplicated().any()
        ):
            raise ValueError(f"Incomplete {split} predictions")
        column = "phq8_pred" if regression else "prob_depressed"
        if not np.isfinite(predictions[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {split} predictions")
        metrics = saved(f"metrics_{split}.json")
        if not math.isfinite(metrics["mae" if regression else "roc_auc"]):
            raise ValueError(f"Invalid {split} metrics")
        pd.read_csv(output / f"metrics_{split}.csv")
    if config["bootstrap"]["enabled"]:
        if len(pd.read_csv(output / "bootstrap_dev.csv")) != config["bootstrap"]["iterations"]:
            raise ValueError("Incomplete bootstrap distribution")
        saved("bootstrap_dev_summary.json")
    repeated = config["evaluation"]["repeated_holdout"]
    if repeated["enabled"]:
        seeds = list(range(repeated["seed_start"], repeated["seed_start"] + repeated["repeats"]))
        if pd.read_csv(output / "repeated_holdout_metrics.csv")["seed"].tolist() != seeds:
            raise ValueError("Incomplete repeated holdouts")
        for seed in seeds:
            assignment = pd.read_csv(output / "repeated_splits" / f"seed_{seed:03d}.csv")
            if (
                len(assignment) != summary["train"]["participants"]
                or assignment["participant_id"].duplicated().any()
            ):
                raise ValueError("Incomplete repeated split provenance")
        saved("repeated_holdout_summary.json")
    saved("environment.json")
    (output / "summary.md").read_text()


def _write_status(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _launch(preset, progress_callback):
    with subprocess.Popen(
        [sys.executable, "-m", "daic_foundation_tab.cli", "run", "--config", str(preset)],
        text=True,
        stdout=subprocess.PIPE,
    ) as process:
        try:
            while True:
                try:
                    stdout, _ = process.communicate(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    progress_callback()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout)
    return Path(stdout.strip().splitlines()[-1]).resolve()


def run_matrix(
    config_directory=Path("configs/experiments"),
    status_path=None,
    *,
    verify_only=False,
    model_family="tabpfn35",
    mode="fine_tuning",
):
    if mode not in {"fine_tuning", "frozen_icl"}:
        raise ValueError("mode must be fine_tuning or frozen_icl")
    available_families = ("tabiclv2", "tabpfn35", "kumo_medium")
    if model_family == "all":
        if mode != "frozen_icl":
            raise ValueError("The all-family matrix is available for frozen_icl experiments")
        families = available_families
    else:
        families = (model_family,)
    if any(family not in available_families for family in families):
        raise ValueError("Unsupported model_family")
    if mode == "fine_tuning" and model_family == "tabiclv2":
        raise ValueError("The fine-tuning matrix supports tabpfn35 and kumo_medium")
    method = "ft" if mode == "fine_tuning" else "icl"
    presets = []
    for family in families:
        family_presets = sorted(
            config_directory.glob(f"{family}_{method}_*.yaml"),
            key=lambda path: (not path.stem.endswith("_smoke"), path.name),
        )
        smoke_count, full_count = (
            (4, 12) if mode == "fine_tuning" and family == "tabpfn35" else (2, 6)
        )
        if len(family_presets) != smoke_count + full_count or sum(
            path.stem.endswith("_smoke") for path in family_presets
        ) != smoke_count:
            raise ValueError(
                f"The {family} {mode} matrix requires {smoke_count} smoke and {full_count} full presets"
            )
        presets.extend(family_presets)
    presets.sort(key=lambda path: (not path.stem.endswith("_smoke"), path.name))
    smoke_count = sum(path.stem.endswith("_smoke") for path in presets)
    if status_path is None:
        if mode == "fine_tuning":
            status_path = Path(f"outputs/{model_family}_matrix_status.json")
        elif model_family == "all":
            status_path = Path("outputs/frozen_icl/matrix_status.json")
        else:
            status_path = Path(f"outputs/frozen_icl/{model_family}_matrix_status.json")
    status_path = Path(status_path)
    configs = {preset.name: load_config(preset) for preset in presets}
    if not verify_only:
        require_cuda_device("cuda:0")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(status_path.read_text()) if status_path.exists() else {"presets": {}}
        logger = configure_run_logger(status_path.with_suffix(".log"), "INFO")
        started = time.perf_counter()
        failures = []
        outputs = {}
        for completed, preset in enumerate(presets):
            if completed == smoke_count and failures and not verify_only:
                raise RuntimeError("Smoke verification failed; full experiments were not started")
            config = configs[preset.name]
            attempts = state["presets"].setdefault(preset.name, [])
            logger.info(
                "phase=matrix preset=%s progress=%s/%s presets (%.1f%%) elapsed=%.1fs ETA unavailable",
                preset.name,
                completed,
                len(presets),
                100 * completed / len(presets),
                time.perf_counter() - started,
            )
            if attempts and attempts[-1]["status"] == "complete":
                try:
                    output = Path(attempts[-1]["output_directory"])
                    verify_run(output, config)
                    logger.info("preset=%s status=verified_complete output=%s", preset.name, output)
                    outputs[preset.name] = output
                    continue
                except (OSError, ValueError, KeyError, TypeError) as error:
                    logger.warning("preset=%s saved run requires retry: %s", preset.name, error)
                    if not verify_only:
                        attempts[-1]["status"] = "invalid"
                        attempts[-1]["failure"] = str(error)
            if verify_only:
                failures.append(preset.name)
                continue
            if attempts and attempts[-1]["status"] == "running":
                attempts[-1]["status"] = "interrupted"
                attempts[-1]["failure"] = (
                    "Previous process did not finish; retrying in a fresh directory"
                )
            record = {
                "preset": preset.name,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": 0,
                "output_directory": None,
                "failure": None,
            }
            attempts.append(record)
            _write_status(status_path, state)
            root = Path(config["project"]["output_root"]).resolve()
            before = set(root.iterdir()) if root.exists() else set()
            run_started = time.perf_counter()

            def update_status(record=record, run_started=run_started, root=root, before=before):
                record["elapsed_seconds"] = time.perf_counter() - run_started
                created = (
                    sorted(path for path in set(root.iterdir()) - before if path.is_dir())
                    if root.exists()
                    else []
                )
                if record["output_directory"] is None and len(created) == 1:
                    record["output_directory"] = str(created[0])
                _write_status(status_path, state)

            try:
                with phase(logger, f"running {preset.stem}"):
                    output = _launch(preset, update_status)
                    record["output_directory"] = str(output)
                    verify_run(output, config)
                record["status"] = "complete"
                outputs[preset.name] = output
            except (Exception, KeyboardInterrupt) as error:
                record["status"] = "failed"
                record["failure"] = f"{type(error).__name__}: {error}"
                failures.append(preset.name)
                logger.error("preset=%s status=failed reason=%s", preset.name, record["failure"])
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                record["ended_at"] = datetime.now(UTC).isoformat()
                update_status()
        if failures:
            raise RuntimeError(f"Matrix incomplete: {', '.join(failures)}")
        for root in {output.parent for output in outputs.values()}:
            for collect in (collect_test_evaluations, collect_regression_test_evaluations):
                table = pd.read_csv(collect(root))
                task = (
                    "regression"
                    if collect is collect_regression_test_evaluations
                    else "classification"
                )
                expected = {
                    output.name
                    for name, output in outputs.items()
                    if output.parent == root and configs[name]["experiment"]["task"] == task
                }
                if not expected.issubset(set(table["run"])):
                    raise ValueError("Aggregate test results are incomplete")
        logger.info(
            "phase=matrix status=complete progress=%s/%s presets (100%%) elapsed=%.1fs",
            len(presets), len(presets),
            time.perf_counter() - started,
        )
        return status_path


def main():
    parser = argparse.ArgumentParser(
        description="Run and resume a foundation-model smoke and full matrix"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--model-family", choices=("tabiclv2", "tabpfn35", "kumo_medium", "all"), default="tabpfn35"
    )
    parser.add_argument("--mode", choices=("fine_tuning", "frozen_icl"), default="fine_tuning")
    args = parser.parse_args()
    print(
        run_matrix(
            verify_only=args.verify_only,
            model_family=args.model_family,
            mode=args.mode,
        )
    )


if __name__ == "__main__":
    main()
