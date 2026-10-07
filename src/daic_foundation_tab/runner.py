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
from daic_foundation_tab.data.labels import load_test_ground_truth
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
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    require_cuda_device,
    reset_cuda_peak_memory,
    timed_call,
)
from daic_foundation_tab.tracking.seeds import set_global_seed
from daic_foundation_tab.tracking.test_evaluations import collect_test_evaluations
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
    test_metrics: dict[str, Any] | None,
    bootstrap_summary: dict[str, Any] | None,
    repeated_summary: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> str:
    train = validation["split_statistics"]["train"]
    dev = validation["split_statistics"]["dev"]
    lines = [
        f"# {model_metadata['model_name']} Classification Result",
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
        "- This model is being evaluated in an extreme small-sample regime.",
        "- Fine-tuning is a feasibility result, not a superiority or clinical-use claim.",
    ]
    if bootstrap_summary is not None:
        lines.extend(["", "## Bootstrap", f"Macro F1 CI: {bootstrap_summary['macro_f1']}"])
    if test_metrics is not None:
        lines.extend(
            [
                "",
                "## Test Result",
                f"Macro F1: {test_metrics['macro_f1']:.4f}",
                f"Depressed F1: {test_metrics['depressed_f1']:.4f}",
                f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}",
                f"ROC-AUC: {test_metrics['roc_auc']:.4f}",
                f"PR-AUC: {test_metrics['pr_auc']:.4f}",
            ]
        )
    if repeated_summary is not None:
        lines.extend(["", "## Repeated Holdout", f"Macro F1: {repeated_summary['macro_f1']}"])
    return "\n".join(lines) + "\n"


def _model_for_seed(model_config: dict[str, Any], random_state: int) -> FineTunableClassifier:
    seeded_config = deepcopy(model_config)
    seeded_config["parameters"]["random_state"] = random_state
    return create_model(seeded_config)


def run_experiment(config: dict[str, Any]) -> Path:
    if config["experiment"]["task"] == "regression":
        from daic_foundation_tab.regression_runner import run_regression_experiment

        return run_regression_experiment(config)
    device = require_cuda_device(config["runtime"]["device"])
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    feature_set = config["experiment"]["feature_set"]
    threshold = float(config["evaluation"]["threshold"])
    fine_tuning_config = config["evaluation"]["fine_tuning"]
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
    with phase(logger, "preparing features") as phase_started:
        dataset, feature_build_seconds = timed_call(
            build_or_load_dataset, config,
            progress_callback=lambda completed, total: log_progress(
                logger, "preparing features", completed, total, phase_started
            ),
        )
        validation = validate_dataset(
            dataset, feature_set, float(config["data"].get("max_exclusion_fraction", 0.05))
        )
        prepared = prepare_fine_tune_splits(
            dataset,
            feature_set,
            float(fine_tuning_config["validation_fraction"]),
            int(fine_tuning_config["validation_seed"]),
        )
    logger.info("participants train=%s dev=%s test=%s", *(validation["split_statistics"][split]["participants"] for split in ("train", "dev", "test")))
    logger.info("feature_count=%s", len(prepared.selection.columns))
    write_config(config, artifacts.path / "config_resolved.yaml")
    environment = {
        **environment_metadata(device),
        "git_commit": _git_commit(),
        "random_seeds": {
            "project": seed,
            "tabpfn" if config["model"]["name"].startswith("tabpfn") else "tabicl": config["model"]["parameters"].get("random_state"),
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
        reset_cuda_peak_memory(device)
        model = _model_for_seed(config["model"], seed)
        logger.info("model=%s checkpoint=%s device=%s", config["model"]["name"], config["model"]["checkpoint_version"], device)
        epochs = int(config["model"]["parameters"].get("epochs", 50))
        with phase(logger, "training") as phase_started:

            def epoch_callback(metrics: dict[str, float]) -> None:
                tracker.record_fine_tuning_epoch(metrics)
                epoch = metrics.get("train/epoch")
                if isinstance(epoch, (int, float)):
                    log_progress(logger, "training", min(int(epoch) + 1, epochs), epochs, phase_started)

            _, fit_seconds = timed_call(
                model.fit,
                prepared.training_x,
                prepared.training_y,
                validation_features=prepared.validation_x,
                validation_target=prepared.validation_y,
                checkpoint_directory=artifacts.path / "checkpoints",
                epoch_callback=epoch_callback,
            )
        finetune_metadata = model.finetune_metadata()
        tracker.record_fine_tuning(
            prepared.training_y,
            prepared.validation_y,
            len(prepared.selection.columns),
            fit_seconds,
            finetune_metadata,
        )
        with phase(logger, "evaluating development"):
            probability, predict_seconds = timed_call(_positive_probability, model, prepared.dev_x)
            prediction = (probability > threshold).astype(int)
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

        test_metrics = None
        if config["evaluation"].get("test_predictions", False):
            with phase(logger, "evaluating test"):
                logger.info("generating official test predictions")
                test_probability, test_predict_seconds = timed_call(
                    _positive_probability, model, prepared.test_x
                )
                test_prediction = (test_probability > threshold).astype(int)
                test_ids = dataset.table.loc[dataset.table["split"] == "test", "participant_id"]
                ground_truth_path = config["data"].get("test_ground_truth")
                test_target = None
                if ground_truth_path:
                    label_path = Path(config["data"]["root"]).expanduser().resolve() / ground_truth_path
                    if label_path.is_file():
                        test_target = load_test_ground_truth(label_path, test_ids)
                        test_metrics = classification_metrics(test_target, test_prediction, test_probability)
                        artifacts.json("metrics_test.json", test_metrics)
                        artifacts.json("decision_threshold.json", {"threshold": threshold})
                        artifacts.csv(
                            "metrics_test.csv",
                            pd.DataFrame([{key: value for key, value in test_metrics.items() if isinstance(value, float)}]),
                        )
                        collect_test_evaluations(artifacts.path.parent)
                        tracker.record_test(test_metrics)
                        logger.info(
                            "test macro_f1=%s balanced_accuracy=%s",
                            test_metrics["macro_f1"],
                            test_metrics["balanced_accuracy"],
                        )
                    else:
                        logger.warning("test ground truth unavailable at %s; skipping test metrics", label_path)
                test_predictions = classification_predictions(
                    test_ids, "test", test_prediction, test_probability, test_target
                )
                artifacts.csv("predictions_test.csv", test_predictions)
                predict_seconds += test_predict_seconds
                logger.info("saved official test predictions for %s participants", len(test_predictions))

        bootstrap_summary = None
        if config["bootstrap"].get("enabled", False):
            with phase(logger, "bootstrapping development") as phase_started:
                bootstrap_distribution, bootstrap_summary = bootstrap_metrics(
                    prepared.dev_y,
                    prediction,
                    probability,
                    int(config["bootstrap"]["iterations"]),
                    float(config["bootstrap"]["confidence"]),
                    int(config["bootstrap"]["random_state"]),
                    progress_callback=lambda completed, total: log_progress(
                        logger, "bootstrapping development", completed, total, phase_started
                    ),
                )
            artifacts.csv("bootstrap_dev.csv", bootstrap_distribution)
            artifacts.json("bootstrap_dev_summary.json", bootstrap_summary)
            tracker.record_bootstrap(bootstrap_summary)

        repeated_summary = None
        repeated_config = config["evaluation"]["repeated_holdout"]
        if repeated_config.get("enabled", False):
            with phase(logger, "repeated holdout") as phase_started:
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
                    threshold=threshold,
                    progress_callback=lambda completed, total: log_progress(
                        logger, "repeated holdout", completed, total, phase_started
                    ),
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
            **cuda_peak_memory(device),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "feature_build_seconds": feature_build_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        model_metadata = model.run_metadata()
        artifacts.json("runtime.json", runtime)
        artifacts.json("model.json", model_metadata)
        artifacts.text(
            "summary.md",
            _summary(model_metadata, validation, metrics, test_metrics, bootstrap_summary, repeated_summary, runtime),
        )
        tracker.complete(model_metadata, environment, runtime)
        return artifacts.path
    except Exception:
        tracker.fail()
        raise
