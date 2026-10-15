from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab import icl_runner, runner
from daic_foundation_tab.matrix import verify_icl_run

from .helpers import write_synthetic_dataset


def test_frozen_icl_runner_saves_provenance_and_loads_test_truth_after_predictions(
    monkeypatch, tmp_path
):
    config = write_synthetic_dataset(tmp_path / "data")
    config["project"]["output_root"] = str(tmp_path / "outputs")
    config["experiment"].update(mode="frozen_icl", name="synthetic_frozen_icl")
    config["model"] = {
        "name": "tabiclv2_icl",
        "checkpoint_version": "tabicl-classifier-v2",
        "parameters": {"n_estimators": 8, "device": "cuda:0", "random_state": 42},
    }
    config["evaluation"]["test_predictions"] = True
    config["evaluation"]["repeated_holdout"]["enabled"] = False
    config["data"]["test_ground_truth"] = "original_labels/ground_truth.csv"
    ground_truth = pd.DataFrame(
        {"Participant_ID": ["500", "501"], "PHQ8_Score": [2, 11]}
    )
    ground_truth.to_csv(Path(config["data"]["root"]) / config["data"]["test_ground_truth"], index=False)
    checkpoint = tmp_path / "tabicl-checkpoint.ckpt"
    checkpoint.write_bytes(b"pretrained checkpoint")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    events = []

    class FakeModel:
        def fit_context(self, features, target):
            events.append("fit_context")
            self.context_rows = len(target)
            self.feature_count = len(features.columns)
            return self

        def predict_proba(self, features):
            events.append("predict")
            return np.tile([0.7, 0.3], (len(features), 1))

        def icl_metadata(self):
            return {
                "context_rows": self.context_rows,
                "feature_count": self.feature_count,
                "n_estimators": 8,
                "weights_before_inference_sha256": "a" * 64,
                "weights_frozen": True,
                "gradients_enabled": False,
                "optimizer_steps": 0,
                "epochs_completed": 0,
                "checkpoint_sha256": checkpoint_digest,
            }

        def run_metadata(self):
            return {
                "model_name": "TabICLv2-ICL",
                "package": "tabicl",
                "package_version": "2.2.0",
                "checkpoint": config["model"]["checkpoint_version"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_digest,
                "frozen": True,
                "parameters": dict(config["model"]["parameters"]),
            }

    class FakeTracker:
        def __init__(self, artifacts):
            self.artifacts = artifacts

        @classmethod
        def start(cls, config, artifacts, validation, dataset_cache_key):
            return cls(artifacts)

        def record_icl_context(self, *args):
            pass

        def record_development(self, *args):
            pass

        def record_test(self, *args):
            pass

        def complete(self, *args):
            self.artifacts.json(
                "wandb_run.json",
                {"status": "disabled", "enabled": False, "mode": "disabled"},
            )

        def fail(self):
            raise AssertionError("Unexpected frozen ICL runner failure")

    original_load_test_ground_truth = icl_runner.load_test_ground_truth

    def track_test_truth(*args, **kwargs):
        events.append("load_test_ground_truth")
        return original_load_test_ground_truth(*args, **kwargs)

    monkeypatch.setattr(icl_runner, "WandbTracker", FakeTracker)
    monkeypatch.setattr(icl_runner, "create_model", lambda _: FakeModel())
    monkeypatch.setattr(icl_runner, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(icl_runner, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        icl_runner,
        "cuda_peak_memory",
        lambda device: {"peak_cuda_allocated_mb": 1.0, "peak_cuda_reserved_mb": 2.0},
    )
    monkeypatch.setattr(icl_runner, "environment_metadata", lambda device: {"cuda_device": device})
    monkeypatch.setattr(icl_runner, "load_test_ground_truth", track_test_truth)

    output = runner.run_experiment(config)

    assert events[0] == "fit_context"
    assert events.index("predict") < events.index("load_test_ground_truth")
    saved_metadata = json.loads((output / "icl_metadata.json").read_text())
    assert saved_metadata["weights_after_inference_sha256"] == "a" * 64
    assert len(pd.read_csv(output / "predictions_test.csv")) == 2
    verify_icl_run(output, config)
