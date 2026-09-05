from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daic_foundation_tab.config import write_config
from daic_foundation_tab.data import build_or_load_dataset
from daic_foundation_tab.data.validation import (
    fit_feature_selection,
    prepare_splits,
    validate_dataset,
)
from daic_foundation_tab.evaluation.bootstrap import bootstrap_metrics
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.predictions import classification_predictions
from daic_foundation_tab.evaluation.repeated_holdout import (
    repeated_holdout,
    repeated_holdout_summary,
)
from daic_foundation_tab.models import create_model
from daic_foundation_tab.models.base import TabularClassifier
from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.environment import environment_metadata
from daic_foundation_tab.tracking.logging import configure_run_logger
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    reset_cuda_peak_memory,
    timed_call,
)


def _positive_probability(model: TabularClassifier, features: pd.DataFrame) -> np.ndarray:
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


def _frozen_config_hash(config: dict[str, Any]) -> str:
    snapshot = {key: value for key, value in config.items() if key != "_config_path"}
    evaluation = dict(snapshot["evaluation"])
    evaluation.pop("test_predictions", None)
    evaluation.pop("frozen_config_sha256", None)
    snapshot["evaluation"] = evaluation
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()


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
        "# TabICLv2 DAIC-WOZ Result",
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
        "- TabICLv2 is being evaluated in an extreme small-sample regime.",
        "- Phase 1 is a feasibility and pipeline-validation result, not a superiority or clinical-use claim.",
    ]
    if bootstrap_summary is not None:
        lines.extend(["", "## Bootstrap", f"Macro F1 CI: {bootstrap_summary['macro_f1']}"])
    if repeated_summary is not None:
        lines.extend(["", "## Repeated Holdout", f"Macro F1: {repeated_summary['macro_f1']}"])
    return "\n".join(lines) + "\n"


def run_experiment(config: dict[str, Any]) -> Path:
    started = time.perf_counter()
    dataset = build_or_load_dataset(config)
    feature_set = config["experiment"]["feature_set"]
    validation = validate_dataset(
        dataset, feature_set, float(config["data"].get("max_exclusion_fraction", 0.05))
    )
    prepared = prepare_splits(dataset, feature_set)
    artifacts = RunArtifacts(
        config["project"]["output_root"],
        config["model"]["name"],
        feature_set,
        int(config["project"]["seed"]),
    )
    logger = configure_run_logger(artifacts.path / "run.log", config["logging"]["level"])
    logger.info("run_id=%s", artifacts.path.name)
    logger.info("feature_set=%s", feature_set)
    logger.info("participants train=%s dev=%s test=%s", *(validation["split_statistics"][split]["participants"] for split in ("train", "dev", "test")))
    logger.info("feature_count=%s", len(prepared.selection.columns))
    write_config(config, artifacts.path / "config_resolved.yaml")
    artifacts.json("environment.json", {**environment_metadata(), "git_commit": _git_commit()})
    artifacts.json("dataset_summary.json", validation["split_statistics"])
    artifacts.json("dataset_inventory.json", dataset.inventory)
    artifacts.json("validation_report.json", validation)
    artifacts.csv("feature_manifest.csv", dataset.manifest)
    artifacts.csv("participant_reconciliation.csv", dataset.reconciliation)
    artifacts.csv(
        "dropped_features.csv",
        pd.DataFrame(
            [
                *( {"column": column, "reason": "all_nan"} for column in prepared.selection.dropped_all_nan ),
                *( {"column": column, "reason": "constant"} for column in prepared.selection.dropped_constant ),
            ]
        ),
    )

    device = reset_cuda_peak_memory()
    model = create_model(config["model"])
    logger.info("model=%s checkpoint=%s device=%s", config["model"]["name"], config["model"]["checkpoint_version"], device)
    _, fit_seconds = timed_call(model.fit, prepared.train_x, prepared.train_y)
    prediction, predict_seconds = timed_call(model.predict, prepared.dev_x)
    probability, probability_seconds = timed_call(_positive_probability, model, prepared.dev_x)
    metrics = classification_metrics(prepared.dev_y, prediction, probability)
    logger.info("development macro_f1=%s balanced_accuracy=%s", metrics["macro_f1"], metrics["balanced_accuracy"])
    dev_ids = dataset.table.loc[dataset.table["split"] == "dev", "participant_id"]
    predictions = classification_predictions(dev_ids, "dev", prediction, probability, prepared.dev_y)
    artifacts.csv("predictions_dev.csv", predictions)
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

    repeated_summary = None
    repeated_config = config["evaluation"]["repeated_holdout"]
    if repeated_config.get("enabled", False):
        train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"]
        repeated_metrics, assignments = repeated_holdout(
            prepared.train_x,
            prepared.train_y,
            train_ids,
            int(repeated_config["repeats"]),
            float(repeated_config["validation_fraction"]),
            int(repeated_config["seed_start"]),
            lambda: create_model(config["model"]),
        )
        repeated_summary = repeated_holdout_summary(repeated_metrics)
        artifacts.csv("repeated_holdout_metrics.csv", repeated_metrics)
        artifacts.json("repeated_holdout_summary.json", repeated_summary)
        split_directory = artifacts.path / "repeated_splits"
        split_directory.mkdir()
        for seed, assignment in assignments.items():
            assignment.to_csv(split_directory / f"seed_{seed:03d}.csv", index=False)

    if config["evaluation"].get("test_predictions", False):
        frozen_hash = config["evaluation"].get("frozen_config_sha256")
        if not frozen_hash:
            raise ValueError("test_predictions requires evaluation.frozen_config_sha256")
        actual_hash = _frozen_config_hash(config)
        if frozen_hash != actual_hash:
            raise ValueError("frozen_config_sha256 does not match the resolved configuration")
        context_x = pd.concat([prepared.train_x, prepared.dev_x], ignore_index=True)
        context_y = pd.concat([prepared.train_y, prepared.dev_y], ignore_index=True)
        selection = fit_feature_selection(context_x)
        final_model = create_model(config["model"])
        final_model.fit(context_x.reindex(columns=selection.columns), context_y)
        test_prediction = final_model.predict(prepared.test_x.reindex(columns=selection.columns))
        test_probability = _positive_probability(final_model, prepared.test_x.reindex(columns=selection.columns))
        test_ids = dataset.table.loc[dataset.table["split"] == "test", "participant_id"]
        artifacts.csv(
            "predictions_test.csv",
            classification_predictions(test_ids, "test", test_prediction, test_probability),
        )

    runtime = {
        "device": device,
        **cuda_peak_memory(),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds + probability_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    artifacts.json("runtime.json", runtime)
    artifacts.json("model.json", model.run_metadata())
    artifacts.text(
        "summary.md",
        _summary(model.run_metadata(), validation, metrics, bootstrap_summary, repeated_summary, runtime),
    )
    return artifacts.path
