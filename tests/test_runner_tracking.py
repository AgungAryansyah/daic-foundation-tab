from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from daic_foundation_tab import runner

from .helpers import write_synthetic_dataset


class _FakeModel:
    def fit(self, features, target, *, validation_features, validation_target, checkpoint_directory):
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        (checkpoint_directory / "best.ckpt").write_bytes(b"checkpoint")
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

    def record_development(self, *args):
        self.calls.append(("development", args))

    def record_bootstrap(self, *args):
        self.calls.append(("bootstrap", args))

    def record_repeated_holdout(self, *args):
        self.calls.append(("repeated_holdout", args))

    def complete(self, *args):
        self.calls.append(("complete", args))

    def fail(self):
        self.calls.append(("fail",))


def test_runner_records_fine_tuning_lifecycle(monkeypatch, tmp_path) -> None:
    _FakeTracker.instances.clear()
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic.yaml"
    monkeypatch.setattr(runner, "WandbTracker", _FakeTracker)
    monkeypatch.setattr(runner, "_model_for_seed", lambda *_: _FakeModel())

    output = runner.run_experiment(config)

    tracker = _FakeTracker.instances[0]
    assert output.is_dir()
    assert [call[0] for call in tracker.calls] == [
        "start",
        "fine_tuning",
        "development",
        "complete",
    ]
    assert (output / "finetune_metadata.json").is_file()
    assert (output / "predictions_dev.csv").is_file()


def test_runner_marks_wandb_failed_when_fine_tuning_errors(monkeypatch, tmp_path) -> None:
    _FakeTracker.instances.clear()
    config = write_synthetic_dataset(tmp_path / "data")
    config["_config_path"] = "synthetic.yaml"
    monkeypatch.setattr(runner, "WandbTracker", _FakeTracker)
    monkeypatch.setattr(runner, "_model_for_seed", lambda *_: _FailingModel())

    with pytest.raises(RuntimeError, match="fine-tuning failed"):
        runner.run_experiment(config)

    assert [call[0] for call in _FakeTracker.instances[0].calls] == ["start", "fail"]
