from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab import runner
from daic_foundation_tab.tracking.runtime import CudaRequirementError

from .helpers import write_synthetic_dataset


class _FakeModel:
    def fit(
        self,
        features,
        target,
        *,
        validation_features,
        validation_target,
        checkpoint_directory,
        epoch_callback=None,
    ):
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        (checkpoint_directory / "best.ckpt").write_bytes(b"checkpoint")
        if epoch_callback is not None:
            epoch_callback({"train/epoch": 0, "train/mean_loss": 0.8, "val/roc_auc": 0.75})
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.7, 0.3], (len(features), 1))

    def finetune_metadata(self):
        return {
            "selection_metric": "roc_auc",
            "best_validation_metric": 0.75,
            "checkpoint_sha256": "checkpoint-hash",
        }

    def run_metadata(self):
        return {"model_name": "FakeModel", "checkpoint": "fake"}


class _FailingModel(_FakeModel):
    def fit(self, *args, **kwargs):
        raise RuntimeError("fine-tuning failed")


class _FakeTracker:
    instances: ClassVar[list[_FakeTracker]] = []

    def __init__(self) -> None:
        self.calls = []
        self.instances.append(self)

    @classmethod
    def start(cls, config, artifacts, validation, dataset_cache_key):
        tracker = cls()
        tracker.calls.append(("start", dataset_cache_key, validation["feature_set"]))
        return tracker

    def record_fine_tuning(self, *args):
        self.calls.append(("fine_tuning", args))

    def record_fine_tuning_epoch(self, *args):
        self.calls.append(("fine_tuning_epoch", args))

    def record_development(self, *args):
        self.calls.append(("development", args))

    def record_test(self, *args):
        self.calls.append(("test", args))

    def record_bootstrap(self, *args):
        self.calls.append(("bootstrap", args))

    def record_repeated_holdout(self, *args):
        self.calls.append(("repeated_holdout", args))

    def complete(self, *args):
        self.calls.append(("complete", args))

    def fail(self):
        self.calls.append(("fail",))


def _mock_gpu_runtime(monkeypatch) -> None:
    monkeypatch.setattr(runner, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(runner, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        runner,
        "cuda_peak_memory",
        lambda device: {"peak_cuda_allocated_mb": 1.0, "peak_cuda_reserved_mb": 2.0},
    )
    monkeypatch.setattr(runner, "environment_metadata", lambda device: {"cuda_device": device})


def test_runner_records_fine_tuning_lifecycle(monkeypatch, tmp_path) -> None:
    _FakeTracker.instances.clear()
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic.yaml"
    config["evaluation"]["test_predictions"] = True
    config["data"]["test_ground_truth"] = "original_labels/missing.csv"
    _mock_gpu_runtime(monkeypatch)
    monkeypatch.setattr(runner, "WandbTracker", _FakeTracker)
    monkeypatch.setattr(runner, "_model_for_seed", lambda *_: _FakeModel())

    output = runner.run_experiment(config)

    tracker = _FakeTracker.instances[0]
    assert output.is_dir()
    assert [call[0] for call in tracker.calls] == [
        "start",
        "fine_tuning_epoch",
        "fine_tuning",
        "development",
        "complete",
    ]
    assert (output / "finetune_metadata.json").is_file()
    assert (output / "predictions_dev.csv").is_file()
    test_predictions = pd.read_csv(output / "predictions_test.csv")
    assert test_predictions["participant_id"].tolist() == [500, 501]
    assert test_predictions["split"].tolist() == ["test", "test"]
    assert test_predictions["y_true"].isna().all()
    assert test_predictions["y_pred"].tolist() == [0, 0]
    assert test_predictions["prob_non_depressed"].tolist() == [0.7, 0.7]
    assert test_predictions["prob_depressed"].tolist() == [0.3, 0.3]
    assert not (output / "metrics_test.json").exists()
    assert not (output.parent / "test_evaluations.csv").exists()


def test_runner_scores_test_after_training_when_ground_truth_is_configured(
    monkeypatch, tmp_path
) -> None:
    _FakeTracker.instances.clear()
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic.yaml"
    config["evaluation"]["test_predictions"] = True
    config["evaluation"]["threshold"] = 0.25
    config["data"]["test_ground_truth"] = "original_labels/full_test_split.csv"
    pd.DataFrame(
        {
            "Participant_ID": [501, 500],
            "PHQ_Binary": [1, 0],
            "PHQ_Score": [12, 3],
        }
    ).to_csv(tmp_path / "data" / "original_labels" / "full_test_split.csv", index=False)
    _mock_gpu_runtime(monkeypatch)
    monkeypatch.setattr(runner, "WandbTracker", _FakeTracker)
    monkeypatch.setattr(runner, "_model_for_seed", lambda *_: _FakeModel())

    output = runner.run_experiment(config)

    predictions = pd.read_csv(output / "predictions_test.csv")
    assert predictions["y_true"].tolist() == [0, 1]
    assert predictions["y_pred"].tolist() == [1, 1]
    assert predictions["prob_depressed"].tolist() == [0.3, 0.3]
    assert (output / "metrics_test.json").is_file()
    assert (output / "metrics_test.csv").is_file()
    assert '"threshold": 0.25' in (output / "decision_threshold.json").read_text(encoding="utf-8")
    comparison = pd.read_csv(output.parent / "test_evaluations.csv")
    assert comparison["run"].tolist() == [output.name]
    assert comparison["feature_set"].tolist() == ["audio_visual"]
    assert comparison["threshold"].tolist() == [0.25]
    assert comparison.loc[0, "macro_f1"] == pd.read_csv(output / "metrics_test.csv").loc[0, "macro_f1"]
    assert "## Test Result" in (output / "summary.md").read_text()
    assert [call[0] for call in _FakeTracker.instances[0].calls] == [
        "start",
        "fine_tuning_epoch",
        "fine_tuning",
        "development",
        "test",
        "complete",
    ]


def test_runner_marks_wandb_failed_when_fine_tuning_errors(monkeypatch, tmp_path) -> None:
    _FakeTracker.instances.clear()
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic.yaml"
    _mock_gpu_runtime(monkeypatch)
    monkeypatch.setattr(runner, "WandbTracker", _FakeTracker)
    monkeypatch.setattr(runner, "_model_for_seed", lambda *_: _FailingModel())

    with pytest.raises(RuntimeError, match="fine-tuning failed"):
        runner.run_experiment(config)

    assert [call[0] for call in _FakeTracker.instances[0].calls] == ["start", "fail"]


def test_runner_rejects_cpu_before_building_dataset(monkeypatch, tmp_path) -> None:
    config = write_synthetic_dataset(tmp_path / "data")
    built = False

    def fail_preflight(device: str) -> str:
        raise CudaRequirementError(f"CUDA unavailable for {device}")

    def unexpected_build(config):
        nonlocal built
        built = True
        raise AssertionError("dataset build must not run")

    monkeypatch.setattr(runner, "require_cuda_device", fail_preflight)
    monkeypatch.setattr(runner, "build_or_load_dataset", unexpected_build)

    with pytest.raises(CudaRequirementError, match="CUDA unavailable"):
        runner.run_experiment(config)

    assert not built
