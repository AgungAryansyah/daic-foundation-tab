from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
import pandas as pd

from daic_foundation_tab import regression_runner, runner

from .helpers import write_synthetic_dataset


class _FakeRegressor:
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
            epoch_callback({"train/epoch": 0, "val/mae": 2.0})
        return self

    def predict(self, features):
        return np.full(len(features), 11.0)

    def finetune_metadata(self):
        return {
            "selection_metric": "mae",
            "best_validation_metric": 2.0,
            "checkpoint_sha256": "checkpoint-hash",
        }

    def run_metadata(self):
        return {"model_name": "FakeRegressor", "checkpoint": "fake"}


class _FakeTracker:
    instances: ClassVar[list[_FakeTracker]] = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    @classmethod
    def start(cls, *args):
        return cls()

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


def _config(tmp_path):
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic_regression.yaml"
    config["experiment"]["task"] = "regression"
    config["model"]["name"] = "tabiclv2_ft_regressor"
    config["model"]["parameters"]["eval_metric"] = "mae"
    config["model"]["parameters"]["epochs"] = 2
    config["evaluation"]["test_predictions"] = True
    config["evaluation"]["repeated_holdout"] = {
        "enabled": True,
        "repeats": 2,
        "validation_fraction": 0.25,
        "seed_start": 0,
    }
    config["bootstrap"] = {
        "enabled": True,
        "iterations": 10,
        "confidence": 0.95,
        "random_state": 42,
    }
    config["data"]["test_ground_truth"] = "original_labels/full_test_split.csv"
    pd.DataFrame({"Participant_ID": [501, 500], "PHQ_Score": [12, 3]}).to_csv(
        tmp_path / "data" / "original_labels" / "full_test_split.csv", index=False
    )
    return config


def test_regression_runner_scores_after_prediction_and_writes_full_workflow(
    monkeypatch, tmp_path
) -> None:
    _FakeTracker.instances.clear()
    config = _config(tmp_path)
    monkeypatch.setattr(regression_runner, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(regression_runner, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        regression_runner, "cuda_peak_memory", lambda device: {"peak_cuda_allocated_mb": 1.0}
    )
    monkeypatch.setattr(
        regression_runner, "environment_metadata", lambda device: {"cuda_device": device}
    )
    monkeypatch.setattr(regression_runner, "_model_for_seed", lambda *_: _FakeRegressor())
    monkeypatch.setattr(regression_runner, "WandbTracker", _FakeTracker)
    events = []
    original_load = regression_runner.load_test_ground_truth

    def load_after_prediction(*args, **kwargs):
        events.append("load_test_truth")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(regression_runner, "load_test_ground_truth", load_after_prediction)
    original_predict = _FakeRegressor.predict

    def record_prediction(self, features):
        events.append("predict")
        return original_predict(self, features)

    monkeypatch.setattr(_FakeRegressor, "predict", record_prediction)

    output = runner.run_experiment(config)

    assert events[:3] == ["predict", "predict", "load_test_truth"]
    predictions = pd.read_csv(output / "predictions_test.csv")
    assert predictions["phq8_true"].tolist() == [3.0, 12.0]
    assert predictions["phq8_pred"].tolist() == [11.0, 11.0]
    assert predictions["depressed_pred"].tolist() == [1, 1]
    metrics = json.loads((output / "metrics_test.json").read_text(encoding="utf-8"))
    assert metrics["mae"] == 4.5
    assert metrics["derived_cutoff"] == 10
    assert pd.read_csv(output.parent / "regression_test_evaluations.csv")["mae"].tolist() == [4.5]
    assert not (output.parent / "test_evaluations.csv").exists()
    assert (output / "bootstrap_dev_summary.json").is_file()
    assert pd.read_csv(output / "repeated_holdout_metrics.csv")["seed"].tolist() == [0, 1]
    assert len(list((output / "repeated_splits").glob("*.csv"))) == 2
    assert "phase=training" in (output / "run.log").read_text(encoding="utf-8")
    assert "## Test Result" in (output / "summary.md").read_text(encoding="utf-8")
    calls = [name for name, _ in _FakeTracker.instances[0].calls]
    assert calls.count("record_test") == 1
    assert calls.count("record_repeated_holdout") == 1
    assert calls[-1] == "complete"


def test_regression_runner_keeps_test_targets_hidden_without_ground_truth(
    monkeypatch, tmp_path
) -> None:
    config = _config(tmp_path)
    config["data"]["test_ground_truth"] = None
    config["evaluation"]["repeated_holdout"]["enabled"] = False
    config["bootstrap"]["enabled"] = False
    monkeypatch.setattr(regression_runner, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(regression_runner, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        regression_runner, "cuda_peak_memory", lambda device: {"peak_cuda_allocated_mb": 1.0}
    )
    monkeypatch.setattr(
        regression_runner, "environment_metadata", lambda device: {"cuda_device": device}
    )
    monkeypatch.setattr(regression_runner, "_model_for_seed", lambda *_: _FakeRegressor())
    monkeypatch.setattr(regression_runner, "WandbTracker", _FakeTracker)

    output = runner.run_experiment(config)

    assert pd.read_csv(output / "predictions_test.csv")["phq8_true"].isna().all()
    assert not (output / "metrics_test.json").exists()
    assert not (output.parent / "regression_test_evaluations.csv").exists()
