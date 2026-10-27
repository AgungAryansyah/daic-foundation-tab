from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab.config import write_config
from daic_foundation_tab.data.labels import load_test_ground_truth
from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.data.transcripts import load_transcripts
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.predictions import classification_predictions
from daic_foundation_tab.evaluation.regression import regression_metrics, regression_predictions
from daic_foundation_tab.models.registry import create_text_model
from daic_foundation_tab.models.text_prompt import select_demonstrations
from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.environment import environment_metadata
from daic_foundation_tab.tracking.lifecycle import RunRecord, record_dataset
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase
from daic_foundation_tab.tracking.runtime import (
    cuda_peak_memory,
    require_cuda_device,
    reset_cuda_peak_memory,
)
from daic_foundation_tab.tracking.seeds import set_global_seed
from daic_foundation_tab.tracking.text_wandb import TextTracker


def run_text_experiment(config: dict) -> Path:
    artifacts = RunArtifacts(
        config["project"]["output_root"],
        config["model"]["name"],
        "transcript",
        config["project"]["seed"],
        resume_path=config.get("_resume_directory"),
    )
    old_record = artifacts.path / "study_run.json"
    if config.get("_resume_directory"):
        if not old_record.is_file():
            raise ValueError("Resume requires a recorded text run")
        if json.loads(old_record.read_text())["status"] == "completed":
            raise ValueError("The run is already completed")
    record = RunRecord(artifacts, config)
    logger = configure_run_logger(
        artifacts.path / "run.log", config.get("logging", {}).get("level", "INFO")
    )
    tracker = None
    try:
        with record.lifecycle():
            started = time.perf_counter()
            device = require_cuda_device(config["runtime"]["device"])
            set_global_seed(config["project"]["seed"])
            reset_cuda_peak_memory(device)
            environment = environment_metadata(device)
            from daic_foundation_tab.runner import _git_commit

            environment["git_commit"] = _git_commit()
            artifacts.json("environment.json", environment)
            with phase(logger, "preparing transcripts") as phase_started:
                dataset = load_transcripts(
                    config,
                    lambda n, total: log_progress(
                        logger, "preparing transcripts", n, total, phase_started
                    ),
                )
            record_dataset(artifacts, config, dataset)
            mode = config["experiment"].get("mode", "fine_tuning")
            training = dataset.examples("fit", labeled=True)
            validation = dataset.examples("selection", labeled=mode == "fine_tuning")
            demonstrations = (
                select_demonstrations(
                    training,
                    config["model"]["parameters"].get("demonstrations", 0),
                    config["project"]["seed"],
                )
                if mode == "frozen_icl"
                else []
            )
            artifacts.csv(
                "demonstrations.csv",
                pd.DataFrame({"participant_id": [item.participant_id for item in demonstrations]}),
            )
            with phase(logger, "loading model"):
                model = create_text_model(config)
                model.prepare(demonstrations)
            metadata = model.run_metadata()
            identity_config = {
                key: value for key, value in config.items() if not key.startswith("_")
            }
            identity_config["model"] = {
                **config["model"],
                "revision": metadata["checkpoint_revision"],
            }
            identity = fingerprint(
                {
                    "config": identity_config,
                    "dataset": dataset.cache_key,
                    "checkpoint_revision": metadata["checkpoint_revision"],
                }
            )
            identity_path = artifacts.path / "run_identity.json"
            if config.get("_resume_directory"):
                if (
                    not identity_path.is_file()
                    or json.loads(identity_path.read_text())["identity"] != identity
                ):
                    raise ValueError("Run resume configuration, model, or input identity mismatch")
            else:
                artifacts.json("run_identity.json", {"identity": identity})
            resolved = {
                **identity_config,
                "model": {**config["model"], "revision": metadata["checkpoint_revision"]},
            }
            write_config(resolved, artifacts.path / "config_resolved.yaml")
            record.update(
                model=metadata,
                source_revision=environment["git_commit"],
                demonstrations_used=len(demonstrations),
                speaker_policy=config["data"]["transcripts"].get(
                    "speaker_policy", "participant_only"
                ),
            )
            with phase(logger, "validating prompts"):
                queries = [*dataset.examples("dev"), *dataset.examples("test")]
                if mode == "fine_tuning":
                    queries.extend([*training, *validation])
                model.preflight(queries)
            tracker = TextTracker(config, artifacts, metadata)
            fit_seconds = 0.0
            if mode == "fine_tuning":
                with phase(logger, "training adapters") as phase_started:

                    def training_progress(completed, total):
                        log_progress(logger, "training adapters", completed, total, phase_started)
                        record.update(completed_steps=completed, total_steps=total)

                    def training_epoch(metrics):
                        with (artifacts.path / "training_history.jsonl").open("a") as history:
                            history.write(json.dumps(metrics) + "\n")
                        tracker.metrics("fine_tuning", metrics)

                    checkpoint_directory = artifacts.path / "checkpoints"
                    resume_checkpoint = (
                        bool(config.get("_resume_directory"))
                        and (checkpoint_directory / "latest.json").is_file()
                    )
                    if config.get("_resume_directory") and not resume_checkpoint:
                        logger.info("No complete training checkpoint; restarting adapter training")
                    model.fit(
                        training,
                        validation,
                        checkpoint_directory,
                        training_progress,
                        resume=resume_checkpoint,
                        epoch_callback=training_epoch,
                    )
                fit_seconds = time.perf_counter() - phase_started
            predict_seconds = 0.0
            for split in ("dev", "test"):
                if split == "test" and not config["evaluation"].get("test_predictions", True):
                    continue
                examples = dataset.examples(split)
                predictions = []
                with (
                    phase(logger, f"predicting {split}") as phase_started,
                    (artifacts.path / f"responses_{split}.jsonl").open(
                        "w", encoding="utf-8"
                    ) as responses,
                ):
                    for completed, example in enumerate(examples, start=1):
                        batch = model.predict([example])
                        if len(batch) != 1:
                            raise ValueError("Model must return one prediction per participant")
                        prediction = batch[0]
                        predictions.append(prediction)
                        responses.write(
                            json.dumps(
                                {
                                    "participant_id": example.participant_id,
                                    "raw_answer": prediction.raw_answer,
                                    "prompt_tokens": prediction.prompt_tokens,
                                }
                            )
                            + "\n"
                        )
                        responses.flush()
                        log_progress(
                            logger, f"predicting {split}", completed, len(examples), phase_started
                        )
                predict_seconds += time.perf_counter() - phase_started
                with phase(logger, f"evaluating {split}"):
                    target = None
                    if split == "dev":
                        target = pd.Series(
                            [item.target for item in dataset.examples("dev", labeled=True)]
                        )
                    else:
                        truth = config["data"].get("test_ground_truth")
                        truth_path = Path(config["data"]["root"]) / truth if truth else None
                        if truth_path and truth_path.is_file():
                            target = load_test_ground_truth(
                                truth_path,
                                pd.Series([item.participant_id for item in examples]),
                                target="binary_depression"
                                if config["experiment"]["task"] == "classification"
                                else "phq8",
                            )
                    ids = pd.Series([item.participant_id for item in examples], dtype=str)
                    values = np.array([item.value for item in predictions])
                    if not np.isfinite(values).all() or len(values) != len(ids):
                        raise ValueError("Predictions must be finite and aligned with participants")
                    if config["experiment"]["task"] == "classification":
                        probability = np.array(
                            [item.probability_positive for item in predictions], dtype=float
                        )
                        if (
                            not np.isfinite(probability).all()
                            or not ((0 <= probability) & (probability <= 1)).all()
                        ):
                            raise ValueError(
                                "Classification probabilities must be finite and within [0,1]"
                            )
                        values = (probability > config["evaluation"]["threshold"]).astype(int)
                        table = classification_predictions(ids, split, values, probability, target)
                        metrics = (
                            classification_metrics(target, values, probability)
                            if target is not None
                            else None
                        )
                    else:
                        table = regression_predictions(ids, split, values, target)
                        metrics = regression_metrics(target, values) if target is not None else None
                    artifacts.csv(f"predictions_{split}.csv", table)
                    if metrics is not None:
                        if config["experiment"]["task"] == "classification":
                            artifacts.json(
                                "decision_threshold.json",
                                {"threshold": config["evaluation"]["threshold"]},
                            )
                        artifacts.json(f"metrics_{split}.json", metrics)
                        artifacts.csv(
                            f"metrics_{split}.csv",
                            pd.DataFrame(
                                [
                                    {
                                        key: value
                                        for key, value in metrics.items()
                                        if isinstance(value, (float, int))
                                    }
                                ]
                            ),
                        )
                        tracker.metrics("development" if split == "dev" else split, metrics)
            metadata = model.run_metadata()
            runtime = {
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": time.perf_counter() - started,
                "device": device,
                **cuda_peak_memory(device),
            }
            artifacts.json("runtime.json", runtime)
            artifacts.json("model_metadata.json", metadata)
            tracker.metrics("runtime", runtime)
            tracker.metrics("model", metadata)
            record.update(model=metadata, runtime=runtime)
            artifacts.text(
                "summary.md",
                f"# Qwen transcript result\n\nTask: {config['experiment']['task']}\n"
                f"Adaptation: {metadata['adaptation']}\n"
                f"Participants: {len(dataset.texts)}\n"
                f"Fit/context pool: {len(training)} participants\n"
                f"Selection pool: {len(validation)} participants\n"
                "\nMetrics and participant predictions are saved beside this summary.\n",
            )
            tracker.finish()
    except BaseException:
        if tracker:
            tracker.finish(failed=True)
        raise
    if (artifacts.path / "metrics_test.csv").is_file():
        from daic_foundation_tab.tracking.test_evaluations import (
            collect_regression_test_evaluations,
            collect_test_evaluations,
        )

        collector = (
            collect_test_evaluations
            if config["experiment"]["task"] == "classification"
            else collect_regression_test_evaluations
        )
        collector(artifacts.path.parent)
    return artifacts.path
