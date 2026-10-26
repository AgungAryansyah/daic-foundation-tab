from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daic_foundation_tab.config import write_config
from daic_foundation_tab.data import build_or_load_dataset
from daic_foundation_tab.data.labels import load_test_ground_truth
from daic_foundation_tab.data.validation import validate_dataset, validate_regression_dataset
from daic_foundation_tab.evaluation.bootstrap import bootstrap_metrics
from daic_foundation_tab.evaluation.fine_tuning import repeated_fine_tune_summary
from daic_foundation_tab.evaluation.icl_holdout import repeated_icl_holdout
from daic_foundation_tab.evaluation.icl_splits import PreparedICLSplits, prepare_icl_splits
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.predictions import classification_predictions
from daic_foundation_tab.evaluation.regression import (
    bootstrap_regression_metrics,
    regression_metrics,
    regression_predictions,
)
from daic_foundation_tab.models import create_model
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
from daic_foundation_tab.tracking.test_evaluations import (
    collect_regression_test_evaluations,
    collect_test_evaluations,
)
from daic_foundation_tab.tracking.wandb import WandbTracker


def _model_for_seed(model_config: dict[str, Any], seed: int):
    from copy import deepcopy

    seeded = deepcopy(model_config)
    seeded["parameters"]["random_state"] = seed
    return create_model(seeded)


def _write_preparation_artifacts(
    artifacts: RunArtifacts, config: dict[str, Any], environment: dict[str, Any],
    validation: dict[str, Any], dataset, prepared: PreparedICLSplits,
) -> None:
    write_config(config, artifacts.path / "config_resolved.yaml")
    artifacts.json("environment.json", environment)
    artifacts.json("dataset_summary.json", validation["split_statistics"])
    artifacts.json("dataset_inventory.json", dataset.inventory)
    artifacts.json("validation_report.json", validation)
    artifacts.csv("feature_manifest.csv", dataset.manifest)
    artifacts.csv("participant_reconciliation.csv", dataset.reconciliation)
    artifacts.csv("icl_split.csv", prepared.assignment)
    artifacts.csv(
        "dropped_features.csv",
        pd.DataFrame(
            [
                *({"column": column, "reason": "all_nan"} for column in prepared.selection.dropped_all_nan),
                *({"column": column, "reason": "constant"} for column in prepared.selection.dropped_constant),
            ]
        ),
    )


def _summary(
    task: str,
    model_metadata: dict[str, Any],
    development: dict[str, Any],
    test: dict[str, Any] | None,
    runtime: dict[str, Any],
    bootstrap: dict[str, Any] | None,
    repeated: dict[str, Any] | None,
) -> str:
    lines = [
        f"# {model_metadata['model_name']} Frozen ICL Result",
        "",
        "## Pretrained model",
        f"Checkpoint: {model_metadata['checkpoint']}",
        f"SHA-256: {model_metadata['checkpoint_sha256']}",
        "",
        "## Development metrics",
    ]
    metrics = ("mae", "rmse", "r2") if task == "regression" else (
        "macro_f1", "depressed_f1", "balanced_accuracy", "roc_auc", "pr_auc"
    )
    lines.extend(f"{key}: {development[key]}" for key in metrics)
    if test is not None:
        lines.extend(["", "## Test metrics"])
        lines.extend(f"{key}: {test[key]}" for key in metrics)
    if bootstrap is not None:
        lines.extend(["", "## Development bootstrap", f"Metrics: {bootstrap}"])
    if repeated is not None:
        lines.extend(["", "## Repeated holdout", f"Metrics: {repeated}"])
    lines.extend(
        [
            "",
            "## Runtime",
            f"Device: {runtime['device']}",
            f"Peak GPU memory (MB): {runtime['peak_cuda_allocated_mb']}",
            f"Total runtime (s): {runtime['total_seconds']:.2f}",
            "",
            "## Inference protocol",
            "The pretrained weights were frozen; no gradients, optimization steps, or epochs were used.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_metric_artifacts(artifacts, prefix: str, metrics: dict[str, Any]) -> None:
    artifacts.json(f"metrics_{prefix}.json", metrics)
    artifacts.csv(
        f"metrics_{prefix}.csv",
        pd.DataFrame(
            [{key: value for key, value in metrics.items() if isinstance(value, float)}]
        ),
    )


def run_icl_experiment(config: dict[str, Any], *, artifacts: RunArtifacts | None = None) -> Path:
    device = require_cuda_device(config["runtime"]["device"])
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    task = config["experiment"]["task"]
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    feature_set = config["experiment"]["feature_set"]
    artifacts = artifacts or RunArtifacts(
        config["project"]["output_root"], config["model"]["name"], feature_set, seed
    )
    logger = configure_run_logger(artifacts.path / "run.log", config["logging"]["level"])
    logger.info("run_id=%s mode=frozen_icl task=%s seed=%s feature_set=%s", artifacts.path.name, task, seed, feature_set)

    validation_function = validate_regression_dataset if task == "regression" else validate_dataset
    with phase(logger, "preparing features") as phase_started:
        dataset, feature_build_seconds = timed_call(
            build_or_load_dataset,
            config,
            progress_callback=lambda completed, total: log_progress(
                logger, "preparing features", completed, total, phase_started
            ),
        )
        validation = validation_function(
            dataset, feature_set, float(config["data"].get("max_exclusion_fraction", 0.05))
        )
        split_config = config["evaluation"]["fine_tuning"]
        prepared = prepare_icl_splits(
            dataset,
            feature_set,
            task,
            float(split_config["validation_fraction"]),
            int(split_config["validation_seed"]),
        )

    model_family = config["model"]["name"].split("_", 1)[0].removesuffix("35")
    environment = {
        **environment_metadata(device),
        "git_commit": _git_commit(),
        "random_seeds": {
            "project": seed,
            model_family: config["model"]["parameters"]["random_state"],
            "bootstrap": config["bootstrap"].get("random_state"),
            "icl_context_partition": split_config["validation_seed"],
            "repeated_holdout_start": config["evaluation"]["repeated_holdout"].get("seed_start"),
        },
    }
    _write_preparation_artifacts(artifacts, config, environment, validation, dataset, prepared)
    from daic_foundation_tab.tracking.lifecycle import record_dataset

    record_dataset(artifacts, config, dataset)
    tracker = WandbTracker.start(config, artifacts, validation, dataset.cache_key)
    try:
        reset_cuda_peak_memory(device)
        model = create_model(config["model"])
        with phase(logger, "preparing frozen ICL context") as phase_started:
            log_progress(
                logger,
                "preparing ICL context",
                0,
                len(prepared.context_x),
                phase_started,
            )
            _, context_seconds = timed_call(model.fit_context, prepared.context_x, prepared.context_y)
            icl_metadata = model.icl_metadata()
            log_progress(
                logger,
                "preparing ICL context",
                len(prepared.context_x),
                len(prepared.context_x),
                phase_started,
            )
        artifacts.json("icl_metadata.json", icl_metadata)
        tracker.record_icl_context(
            prepared.context_y,
            prepared.reserved_y,
            len(prepared.selection.columns),
            context_seconds,
            icl_metadata,
        )

        threshold = float(config["evaluation"]["threshold"]) if task == "classification" else None
        with phase(logger, "evaluating development"):
            dev_ids = dataset.table.loc[dataset.table["split"] == "dev", "participant_id"]
            if task == "regression":
                dev_prediction_raw, dev_seconds = timed_call(model.predict, prepared.dev_x)
                dev_prediction = np.asarray(dev_prediction_raw, dtype=float)
                dev_metrics = regression_metrics(prepared.dev_y, dev_prediction)
                artifacts.csv(
                    "predictions_dev.csv",
                    regression_predictions(dev_ids, "dev", dev_prediction, prepared.dev_y),
                )
            else:
                probability_raw, dev_seconds = timed_call(model.predict_proba, prepared.dev_x)
                dev_probability = np.asarray(probability_raw, dtype=float)[:, 1]
                dev_prediction = (dev_probability > threshold).astype(int)
                dev_metrics = classification_metrics(prepared.dev_y, dev_prediction, dev_probability)
                artifacts.csv(
                    "predictions_dev.csv",
                    classification_predictions(
                        dev_ids, "dev", dev_prediction, dev_probability, prepared.dev_y
                    ),
                )
            _save_metric_artifacts(artifacts, "dev", dev_metrics)
            tracker.record_development(dev_metrics)

        test_metrics = None
        test_seconds = 0.0
        if config["evaluation"].get("test_predictions", False):
            with phase(logger, "evaluating official test"):
                test_ids = dataset.table.loc[dataset.table["split"] == "test", "participant_id"]
                test_target = None
                ground_truth = config["data"].get("test_ground_truth")
                label_path = Path(config["data"]["root"]).expanduser().resolve() / ground_truth if ground_truth else None
                if task == "regression":
                    test_prediction_raw, test_seconds = timed_call(model.predict, prepared.test_x)
                    test_prediction = np.asarray(test_prediction_raw, dtype=float)
                    if label_path is not None and label_path.is_file():
                        test_target = load_test_ground_truth(label_path, test_ids, target="phq8")
                        test_metrics = regression_metrics(test_target, test_prediction)
                        _save_metric_artifacts(artifacts, "test", test_metrics)
                        collect_regression_test_evaluations(artifacts.path.parent)
                        tracker.record_test(test_metrics)
                    artifacts.csv(
                        "predictions_test.csv",
                        regression_predictions(test_ids, "test", test_prediction, test_target),
                    )
                else:
                    probability_raw, test_seconds = timed_call(
                        model.predict_proba, prepared.test_x
                    )
                    test_probability = np.asarray(probability_raw, dtype=float)[:, 1]
                    test_prediction = (test_probability > threshold).astype(int)
                    if label_path is not None and label_path.is_file():
                        test_target = load_test_ground_truth(label_path, test_ids)
                        test_metrics = classification_metrics(test_target, test_prediction, test_probability)
                        _save_metric_artifacts(artifacts, "test", test_metrics)
                        artifacts.json("decision_threshold.json", {"threshold": threshold})
                        collect_test_evaluations(artifacts.path.parent)
                        tracker.record_test(test_metrics)
                    artifacts.csv(
                        "predictions_test.csv",
                        classification_predictions(
                            test_ids, "test", test_prediction, test_probability, test_target
                        ),
                    )

        final_icl_metadata = model.icl_metadata()
        if (
            final_icl_metadata["weights_before_inference_sha256"]
            != icl_metadata["weights_before_inference_sha256"]
            or final_icl_metadata["checkpoint_sha256"] != icl_metadata["checkpoint_sha256"]
        ):
            raise RuntimeError("Frozen ICL weights or checkpoint changed during evaluation")
        icl_metadata["weights_after_inference_sha256"] = final_icl_metadata[
            "weights_before_inference_sha256"
        ]
        artifacts.json("icl_metadata.json", icl_metadata)
        model_metadata = model.run_metadata()
        model.close()

        bootstrap_summary = None
        if config["bootstrap"].get("enabled", False):
            with phase(logger, "bootstrapping development") as phase_started:
                if task == "regression":
                    distribution, bootstrap_summary = bootstrap_regression_metrics(
                        prepared.dev_y,
                        dev_prediction,
                        int(config["bootstrap"]["iterations"]),
                        float(config["bootstrap"]["confidence"]),
                        int(config["bootstrap"]["random_state"]),
                        progress_callback=lambda completed, total: log_progress(
                            logger, "bootstrapping development", completed, total, phase_started
                        ),
                    )
                else:
                    distribution, bootstrap_summary = bootstrap_metrics(
                        prepared.dev_y,
                        dev_prediction,
                        dev_probability,
                        int(config["bootstrap"]["iterations"]),
                        float(config["bootstrap"]["confidence"]),
                        int(config["bootstrap"]["random_state"]),
                        progress_callback=lambda completed, total: log_progress(
                            logger, "bootstrapping development", completed, total, phase_started
                        ),
                    )
            artifacts.csv("bootstrap_dev.csv", distribution)
            artifacts.json("bootstrap_dev_summary.json", bootstrap_summary)
            tracker.record_bootstrap(bootstrap_summary)

        repeated_summary = None
        repeated = config["evaluation"]["repeated_holdout"]
        if repeated.get("enabled", False):
            train_x, train_y = dataset.get_split(
                "train", feature_set, target="phq8" if task == "regression" else "binary_depression"
            )
            train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"]
            with phase(logger, "repeated frozen ICL holdout") as phase_started:
                holdout_metrics, assignments = repeated_icl_holdout(
                    train_x,
                    train_y.astype(float if task == "regression" else int),
                    train_ids,
                    int(repeated["repeats"]),
                    float(repeated["validation_fraction"]),
                    float(split_config["validation_fraction"]),
                    int(repeated["seed_start"]),
                    lambda holdout_seed: _model_for_seed(config["model"], holdout_seed),
                    task=task,
                    threshold=0.5 if threshold is None else threshold,
                    progress_callback=lambda completed, total: log_progress(
                        logger, "repeated holdout", completed, total, phase_started
                    ),
                )
            repeated_summary = repeated_fine_tune_summary(holdout_metrics)
            artifacts.csv("repeated_holdout_metrics.csv", holdout_metrics)
            artifacts.json("repeated_holdout_summary.json", repeated_summary)
            tracker.record_repeated_holdout(holdout_metrics, repeated_summary)
            split_directory = artifacts.path / "repeated_splits"
            split_directory.mkdir()
            for holdout_seed, assignment in assignments.items():
                assignment.to_csv(split_directory / f"seed_{holdout_seed:03d}.csv", index=False)

        ended_at = datetime.now(UTC)
        runtime = {
            "device": device,
            **cuda_peak_memory(device),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "feature_build_seconds": feature_build_seconds,
            "context_setup_seconds": context_seconds,
            "fit_seconds": 0.0,
            "predict_seconds": dev_seconds + test_seconds,
            "epochs_completed": 0,
            "optimizer_steps": 0,
            "total_seconds": time.perf_counter() - started,
        }
        artifacts.json("runtime.json", runtime)
        artifacts.json("model.json", model_metadata)
        artifacts.text(
            "summary.md",
            _summary(task, model_metadata, dev_metrics, test_metrics, runtime, bootstrap_summary, repeated_summary),
        )
        tracker.complete(model_metadata, environment, runtime)
        return artifacts.path
    except Exception:
        tracker.fail()
        raise


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
