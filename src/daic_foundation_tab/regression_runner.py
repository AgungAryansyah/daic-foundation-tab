from __future__ import annotations

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
from daic_foundation_tab.data.validation import validate_regression_dataset
from daic_foundation_tab.evaluation.fine_tuning import repeated_fine_tune_summary
from daic_foundation_tab.evaluation.regression import (
    bootstrap_regression_metrics,
    regression_metrics,
    regression_predictions,
)
from daic_foundation_tab.evaluation.regression_splits import (
    prepare_regression_splits,
    repeated_regression_holdout,
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
from daic_foundation_tab.tracking.test_evaluations import collect_regression_test_evaluations
from daic_foundation_tab.tracking.wandb import WandbTracker


def _model_for_seed(model_config: dict[str, Any], seed: int):
    seeded = deepcopy(model_config)
    seeded["parameters"]["random_state"] = seed
    return create_model(seeded)


def _summary(
    model_metadata: dict[str, Any],
    validation: dict[str, Any],
    dev_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
    bootstrap_summary: dict[str, Any] | None,
    repeated_summary: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> str:
    train = validation["split_statistics"]["train"]
    dev = validation["split_statistics"]["dev"]
    lines = [
        f"# {model_metadata['model_name']} PHQ-8 Regression Result",
        "",
        "## Dataset",
        f"Train participants: {train['participants']}",
        f"Development participants: {dev['participants']}",
        "",
        "## Model",
        model_metadata["model_name"],
        f"Checkpoint: {model_metadata['checkpoint']}",
        "",
        "## Development Result",
        f"MAE: {dev_metrics['mae']:.4f}",
        f"RMSE: {dev_metrics['rmse']:.4f}",
        f"R²: {dev_metrics['r2']:.4f}",
        f"Derived depressed F1 (PHQ-8 ≥ 10): {dev_metrics['derived_depressed_f1']:.4f}",
    ]
    if test_metrics is not None:
        lines.extend(
            [
                "",
                "## Test Result",
                f"MAE: {test_metrics['mae']:.4f}",
                f"RMSE: {test_metrics['rmse']:.4f}",
                f"R²: {test_metrics['r2']:.4f}",
                f"Derived depressed F1 (PHQ-8 ≥ 10): {test_metrics['derived_depressed_f1']:.4f}",
            ]
        )
    if bootstrap_summary is not None:
        lines.extend(["", "## Bootstrap", f"MAE CI: {bootstrap_summary['mae']}"])
    if repeated_summary is not None:
        lines.extend(["", "## Repeated Holdout", f"MAE: {repeated_summary['mae']}"])
    lines.extend(
        [
            "",
            "## Resources",
            f"Peak GPU memory (MB): {runtime['peak_cuda_allocated_mb']}",
            f"Total runtime (s): {runtime['total_seconds']:.2f}",
            "",
            "## Limitations",
            "- This is a small-sample feasibility experiment, not a clinical-use claim.",
            "- PHQ-8 predictions are raw model outputs and may fall outside 0–24.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_regression_experiment(config: dict[str, Any]) -> Path:
    device = require_cuda_device(config["runtime"]["device"])
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    feature_set = config["experiment"]["feature_set"]
    artifacts = RunArtifacts(
        config["project"]["output_root"], config["model"]["name"], feature_set, seed
    )
    logger = configure_run_logger(artifacts.path / "run.log", config["logging"]["level"])
    logger.info(
        "run_id=%s task=regression seed=%s feature_set=%s", artifacts.path.name, seed, feature_set
    )

    with phase(logger, "preparing features"):
        dataset, feature_build_seconds = timed_call(build_or_load_dataset, config)
        validation = validate_regression_dataset(
            dataset, feature_set, float(config["data"].get("max_exclusion_fraction", 0.05))
        )
        fine_tuning_config = config["evaluation"]["fine_tuning"]
        prepared = prepare_regression_splits(
            dataset,
            feature_set,
            float(fine_tuning_config["validation_fraction"]),
            int(fine_tuning_config["validation_seed"]),
        )
    logger.info(
        "participants train=%s dev=%s test=%s features=%s",
        *(
            validation["split_statistics"][split]["participants"]
            for split in ("train", "dev", "test")
        ),
        len(prepared.selection.columns),
    )
    write_config(config, artifacts.path / "config_resolved.yaml")
    from daic_foundation_tab.runner import _git_commit

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
                *(
                    {"column": column, "reason": "all_nan"}
                    for column in prepared.selection.dropped_all_nan
                ),
                *(
                    {"column": column, "reason": "constant"}
                    for column in prepared.selection.dropped_constant
                ),
            ]
        ),
    )

    tracker = WandbTracker.start(config, artifacts, validation, dataset.cache_key)
    try:
        reset_cuda_peak_memory(device)
        model = _model_for_seed(config["model"], seed)
        epochs = int(config["model"]["parameters"]["epochs"])
        with phase(logger, "training") as phase_started:

            def epoch_callback(metrics: dict[str, float]) -> None:
                tracker.record_fine_tuning_epoch(metrics)
                epoch = metrics.get("train/epoch")
                if isinstance(epoch, (int, float)):
                    log_progress(
                        logger, "training", min(int(epoch) + 1, epochs), epochs, phase_started
                    )

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
        artifacts.json("finetune_metadata.json", finetune_metadata)

        with phase(logger, "evaluating development"):
            prediction, predict_seconds = timed_call(model.predict, prepared.dev_x)
            prediction = np.asarray(prediction, dtype=float)
            dev_metrics = regression_metrics(prepared.dev_y, prediction)
            dev_ids = dataset.table.loc[dataset.table["split"] == "dev", "participant_id"]
            artifacts.csv(
                "predictions_dev.csv",
                regression_predictions(dev_ids, "dev", prediction, prepared.dev_y),
            )
            artifacts.json("metrics_dev.json", dev_metrics)
            artifacts.csv(
                "metrics_dev.csv",
                pd.DataFrame(
                    [{key: value for key, value in dev_metrics.items() if isinstance(value, float)}]
                ),
            )
            tracker.record_development(dev_metrics)
            logger.info("development mae=%.4f rmse=%.4f", dev_metrics["mae"], dev_metrics["rmse"])

        test_metrics = None
        if config["evaluation"].get("test_predictions", False):
            with phase(logger, "evaluating test"):
                test_prediction, test_predict_seconds = timed_call(model.predict, prepared.test_x)
                test_prediction = np.asarray(test_prediction, dtype=float)
                test_ids = dataset.table.loc[dataset.table["split"] == "test", "participant_id"]
                ground_truth_path = config["data"].get("test_ground_truth")
                test_target = None
                if ground_truth_path:
                    label_path = (
                        Path(config["data"]["root"]).expanduser().resolve() / ground_truth_path
                    )
                    if label_path.is_file():
                        test_target = load_test_ground_truth(label_path, test_ids, target="phq8")
                        test_metrics = regression_metrics(test_target, test_prediction)
                        artifacts.json("metrics_test.json", test_metrics)
                        artifacts.csv(
                            "metrics_test.csv",
                            pd.DataFrame(
                                [
                                    {
                                        key: value
                                        for key, value in test_metrics.items()
                                        if isinstance(value, float)
                                    }
                                ]
                            ),
                        )
                        collect_regression_test_evaluations(artifacts.path.parent)
                        tracker.record_test(test_metrics)
                        logger.info(
                            "test mae=%.4f rmse=%.4f", test_metrics["mae"], test_metrics["rmse"]
                        )
                    else:
                        logger.warning(
                            "test ground truth unavailable at %s; skipping test metrics", label_path
                        )
                artifacts.csv(
                    "predictions_test.csv",
                    regression_predictions(test_ids, "test", test_prediction, test_target),
                )
                predict_seconds += test_predict_seconds
                logger.info("test predictions=%s/%s (100%%)", len(test_prediction), len(test_ids))

        bootstrap_summary = None
        if config["bootstrap"].get("enabled", False):
            with phase(logger, "bootstrapping development") as phase_started:
                bootstrap_distribution, bootstrap_summary = bootstrap_regression_metrics(
                    prepared.dev_y,
                    prediction,
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
                train_x, train_y = dataset.get_split("train", feature_set, target="phq8")
                train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"]
                repeated_metrics, assignments = repeated_regression_holdout(
                    train_x,
                    train_y.astype(float),
                    train_ids,
                    int(repeated_config["repeats"]),
                    float(repeated_config["validation_fraction"]),
                    float(fine_tuning_config["validation_fraction"]),
                    int(repeated_config["seed_start"]),
                    lambda repeat_seed: _model_for_seed(config["model"], repeat_seed),
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
            for repeat_seed, assignment in assignments.items():
                assignment.to_csv(split_directory / f"seed_{repeat_seed:03d}.csv", index=False)

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
            _summary(
                model_metadata,
                validation,
                dev_metrics,
                test_metrics,
                bootstrap_summary,
                repeated_summary,
                runtime,
            ),
        )
        tracker.complete(model_metadata, environment, runtime)
        return artifacts.path
    except Exception:
        tracker.fail()
        raise
