from __future__ import annotations

import subprocess
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daic_foundation_tab.config import write_config
from daic_foundation_tab.data import build_or_load_dataset
from daic_foundation_tab.data.validation import validate_dataset
from daic_foundation_tab.evaluation.bootstrap import bootstrap_metrics
from daic_foundation_tab.evaluation.fine_tuning import (
    prepare_fine_tune_splits,
    repeated_fine_tune_holdout,
    repeated_fine_tune_summary,
)
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.predictions import classification_predictions
from daic_foundation_tab.models import create_model
from daic_foundation_tab.models.base import FineTunableClassifier
from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.environment import environment_metadata
from daic_foundation_tab.tracking.logging import configure_run_logger
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    reset_cuda_peak_memory,
    timed_call,
)
from daic_foundation_tab.tracking.seeds import set_global_seed
from daic_foundation_tab.tracking.wandb import WandbTracker


def _positive_probability(model: FineTunableClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier predict_proba must return two probability columns")
    return probabilities[:, 1]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(
    model_metadata: dict[str, Any],
    validation: dict[str, Any],
    metrics: dict[str, Any],
    bootstrap_summary: dict[str, Any] | None,
    repeated_summary: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> str:
    train = validation["split_statistics"]["train"]
    dev = validation["split_statistics"]["dev"]
    lines = [
        "# TabICLv2-FT DAIC-WOZ Result",
        "",
        "## Dataset",
        f"Train participants: {train['participants']}",
        f"Development participants: {dev['participants']}",
        f"Development positive prevalence: {dev['positive_rate']}",
        "",
        "## Model",
        model_metadata["model_name"],
        f"Checkpoint: {model_metadata['checkpoint']}",
        "",
        "## Development Result",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Depressed F1: {metrics['depressed_f1']:.4f}",
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"PR-AUC: {metrics['pr_auc']:.4f}",
        "",
        "## Resources",
        f"Peak GPU memory (MB): {runtime['peak_cuda_allocated_mb']}",
        f"Total runtime (s): {runtime['total_seconds']:.2f}",
        "",
        "## Limitations",
        "- TabICLv2-FT is being evaluated in an extreme small-sample regime.",
        "- Fine-tuning is a feasibility result, not a superiority or clinical-use claim.",
    ]
    if bootstrap_summary is not None:
        lines.extend(["", "## Bootstrap", f"Macro F1 CI: {bootstrap_summary['macro_f1']}"])
    if repeated_summary is not None:
        lines.extend(["", "## Repeated Holdout", f"Macro F1: {repeated_summary['macro_f1']}"])
    return "\n".join(lines) + "\n"


def _model_for_seed(model_config: dict[str, Any], random_state: int) -> FineTunableClassifier:
    seeded_config = deepcopy(model_config)
    seeded_config["parameters"]["random_state"] = random_state
    return create_model(seeded_config)


def run_experiment(config: dict[str, Any]) -> Path:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    dataset, feature_build_seconds = timed_call(build_or_load_dataset, config)
    feature_set = config["experiment"]["feature_set"]
    validation = validate_dataset(
        dataset, feature_set, float(config["data"].get("max_exclusion_fraction", 0.05))
    )
    fine_tuning_config = config["evaluation"]["fine_tuning"]
    prepared = prepare_fine_tune_splits(
        dataset,
        feature_set,
        float(fine_tuning_config["validation_fraction"]),
        int(fine_tuning_config["validation_seed"]),
    )
    artifacts = RunArtifacts(
        config["project"]["output_root"],
        config["model"]["name"],
        feature_set,
        int(config["project"]["seed"]),
    )
    logger = configure_run_logger(artifacts.path / "run.log", config["logging"]["level"])
    logger.info("run_id=%s", artifacts.path.name)
    logger.info("config_path=%s", config["_config_path"])
    logger.info("dataset_root=%s", config["data"]["root"])
    logger.info("feature_set=%s", feature_set)
    logger.info("seed=%s", seed)
    logger.info("participants train=%s dev=%s test=%s", *(validation["split_statistics"][split]["participants"] for split in ("train", "dev", "test")))
    logger.info("feature_count=%s", len(prepared.selection.columns))
    write_config(config, artifacts.path / "config_resolved.yaml")
    environment = {
        **environment_metadata(),
        "git_commit": _git_commit(),
        "random_seeds": {
            "project": seed,
            "tabicl": config["model"]["parameters"].get("random_state"),
            "bootstrap": config["bootstrap"].get("random_state"),
            "fine_tuning_validation": fine_tuning_config["validation_seed"],
            "repeated_holdout_start": config["evaluation"]["repeated_holdout"].get("seed_start"),
        },
    }
    artifacts.json("environment.json", environment)
    artifacts.json("dataset_summary.json", validation["split_statistics"])
    artifacts.json("dataset_inventory.json", dataset.inventory)
    artifacts.json("validation_report.json", validation)
    artifacts.csv("feature_manifest.csv", dataset.manifest)
    artifacts.csv("participant_reconciliation.csv", dataset.reconciliation)
    artifacts.csv("finetune_split.csv", prepared.assignment)
    artifacts.csv(
        "dropped_features.csv",
        pd.DataFrame(
            [
                *( {"column": column, "reason": "all_nan"} for column in prepared.selection.dropped_all_nan ),
                *( {"column": column, "reason": "constant"} for column in prepared.selection.dropped_constant ),
            ]
        ),
    )

    tracker = WandbTracker.start(config, artifacts, validation, dataset.cache_key)
    try:
        device = reset_cuda_peak_memory()
        model = _model_for_seed(config["model"], seed)
        logger.info("model=%s checkpoint=%s device=%s", config["model"]["name"], config["model"]["checkpoint_version"], device)
        _, fit_seconds = timed_call(
            model.fit,
            prepared.training_x,
            prepared.training_y,
            validation_features=prepared.validation_x,
            validation_target=prepared.validation_y,
            checkpoint_directory=artifacts.path / "checkpoints",
        )
        finetune_metadata = model.finetune_metadata()
        tracker.record_fine_tuning(
            prepared.training_y,
            prepared.validation_y,
            len(prepared.selection.columns),
            fit_seconds,
            finetune_metadata,
        )
        prediction, predict_seconds = timed_call(model.predict, prepared.dev_x)
        probability, probability_seconds = timed_call(_positive_probability, model, prepared.dev_x)
        metrics = classification_metrics(prepared.dev_y, prediction, probability)
        tracker.record_development(metrics)
        logger.info("development macro_f1=%s balanced_accuracy=%s", metrics["macro_f1"], metrics["balanced_accuracy"])
        dev_ids = dataset.table.loc[dataset.table["split"] == "dev", "participant_id"]
        predictions = classification_predictions(dev_ids, "dev", prediction, probability, prepared.dev_y)
        artifacts.csv("predictions_dev.csv", predictions)
        artifacts.json("finetune_metadata.json", finetune_metadata)
        artifacts.json("metrics_dev.json", metrics)
        artifacts.csv(
            "metrics_dev.csv",
            pd.DataFrame([{key: value for key, value in metrics.items() if isinstance(value, float)}]),
        )

        bootstrap_summary = None
        if config["bootstrap"].get("enabled", False):
            bootstrap_distribution, bootstrap_summary = bootstrap_metrics(
                prepared.dev_y,
                prediction,
                probability,
                int(config["bootstrap"]["iterations"]),
                float(config["bootstrap"]["confidence"]),
                int(config["bootstrap"]["random_state"]),
            )
            artifacts.csv("bootstrap_dev.csv", bootstrap_distribution)
            artifacts.json("bootstrap_dev_summary.json", bootstrap_summary)
            tracker.record_bootstrap(bootstrap_summary)

        repeated_summary = None
        repeated_config = config["evaluation"]["repeated_holdout"]
        if repeated_config.get("enabled", False):
            train_x, train_y = dataset.get_split("train", feature_set)
            train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"]
            repeated_metrics, assignments = repeated_fine_tune_holdout(
                train_x,
                train_y.astype(int),
                train_ids,
                int(repeated_config["repeats"]),
                float(repeated_config["validation_fraction"]),
                float(fine_tuning_config["validation_fraction"]),
                int(repeated_config["seed_start"]),
                lambda repeat_seed: _model_for_seed(config["model"], repeat_seed),
            )
            repeated_summary = repeated_fine_tune_summary(repeated_metrics)
            artifacts.csv("repeated_holdout_metrics.csv", repeated_metrics)
            artifacts.json("repeated_holdout_summary.json", repeated_summary)
            tracker.record_repeated_holdout(repeated_metrics, repeated_summary)
            split_directory = artifacts.path / "repeated_splits"
            split_directory.mkdir()
            for seed, assignment in assignments.items():
                assignment.to_csv(split_directory / f"seed_{seed:03d}.csv", index=False)

        ended_at = datetime.now(UTC)
        runtime = {
            "device": device,
            **cuda_peak_memory(),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "feature_build_seconds": feature_build_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds + probability_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        model_metadata = model.run_metadata()
        artifacts.json("runtime.json", runtime)
        artifacts.json("model.json", model_metadata)
        artifacts.text(
            "summary.md",
            _summary(model_metadata, validation, metrics, bootstrap_summary, repeated_summary, runtime),
        )
        tracker.complete(model_metadata, environment, runtime)
        return artifacts.path
    except Exception:
        tracker.fail()
        raise
