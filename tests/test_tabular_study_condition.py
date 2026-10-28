import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.tabular_study import condition as execution
from daic_foundation_tab.tabular_study.bundle import prepare_bundle, verify_bundle
from tests.test_tabular_study_bundle import study_input


class FakeModel:
    def __init__(self, config, calls):
        self.config = config
        self.calls = calls

    def fit(
        self,
        features,
        target,
        *,
        validation_features,
        validation_target,
        checkpoint_directory,
        epoch_callback,
    ):
        assert set(features.index).isdisjoint(validation_features.index)
        self.calls.append(("fit", set(features.index), set(validation_features.index)))
        checkpoint_directory.mkdir()
        self.checkpoint = checkpoint_directory / "best.pt"
        self.checkpoint.write_bytes(b"best")
        (checkpoint_directory / "extra.pt").write_bytes(b"extra")
        epoch_callback({"train/epoch": 0, "train/mean_loss": 0.2, "val/roc_auc": 0.5})

    def fit_context(self, features, target):
        self.calls.append(("context", set(features.index), None))

    def predict_proba(self, features):
        self.calls.append(("predict", set(features.index), None))
        values = np.linspace(0.2, 0.8, len(features))
        return np.column_stack([1 - values, values])

    def predict(self, features):
        self.calls.append(("predict", set(features.index), None))
        return np.repeat(10.0, len(features))

    def finetune_metadata(self):
        return {
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
            "selection_metric": "roc_auc",
            "best_validation_metric": 0.5,
        }

    def icl_metadata(self):
        return {"weights_frozen": True, "optimizer_steps": 0}

    def run_metadata(self):
        return {"model_name": self.config["name"], "checkpoint": self.config["checkpoint_version"]}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    bundle = prepare_bundle(study_input(tmp_path), tmp_path / "bundle")
    calls = []
    monkeypatch.setattr(execution, "create_model", lambda config: FakeModel(config, calls))
    monkeypatch.setattr(execution, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(execution, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        execution, "cuda_peak_memory", lambda device: {"peak_cuda_allocated_mb": 12}
    )
    monkeypatch.setattr(execution, "environment_metadata", lambda device: {"gpu_name": "synthetic"})
    monkeypatch.setattr(execution, "_sync", lambda device: None)
    monkeypatch.setattr(execution, "_source_checkpoint_hash", lambda condition, detail: "a" * 64)
    return bundle, calls


@pytest.mark.parametrize("family", ["tabpfn35", "kumo_medium"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cv_evaluates_only_outer_fold_with_fresh_models(prepared, tmp_path, family, mode, task):
    bundle, calls = prepared
    study = verify_bundle(bundle)
    condition = next(
        item
        for item in study["conditions"]["cv"]
        if item["family"] == family
        and item["mode"] == mode
        and item["task"] == task
        and item["modality"] == "audio_visual"
        and item["fold"] == 1
    )
    output = execution.run_condition(
        bundle, condition, {"output_root": str(tmp_path / "runs")}, "attempt-1"
    )
    record = json.loads((output / "study_run.json").read_text())
    roles = json.loads((output / "fold_assignments.json").read_text())["roles"]
    assert record["status"] == "completed" and record["fold"] == 1
    assert calls[0][0] == ("fit" if mode == "fine_tuning" else "context")
    assert calls[0][1] == set(roles["fit"])
    assert calls[1][1] == set(roles["evaluation"])
    assert calls[0][1].isdisjoint(calls[1][1])
    assert set(pd.read_csv(output / "predictions_cv.csv")["split"]) == {"cv"}
    assert not (output / "predictions_dev.csv").exists()
    assert not (output / "predictions_test.csv").exists()
    if mode == "fine_tuning":
        assert calls[0][2] == set(roles["selection"])
        assert [p.name for p in (output / "checkpoints").iterdir()] == ["best.pt"]


def test_final_stage_scores_test_only_after_prediction(prepared, tmp_path, monkeypatch):
    bundle, calls = prepared
    condition = verify_bundle(bundle)["conditions"]["final"][0]
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"Participant_ID": [500, 501], "PHQ8_Score": [3, 14]}).to_csv(truth, index=False)
    read_truth = execution.load_test_ground_truth

    def checked_truth(*args, **kwargs):
        assert [item[0] for item in calls] == ["context", "predict", "predict"]
        return read_truth(*args, **kwargs)

    monkeypatch.setattr(execution, "load_test_ground_truth", checked_truth)
    output = execution.run_condition(
        bundle,
        condition,
        {"output_root": str(tmp_path / "runs"), "test_ground_truth": str(truth)},
        "attempt-1",
    )
    assert [item[0] for item in calls] == ["context", "predict", "predict"]
    assert (output / "metrics_dev.json").is_file()
    assert (output / "metrics_test.json").is_file()
    assert not (output / "metrics_cv.json").exists()


def test_dependency_check_rejects_version_changes():
    versions = execution._dependency_versions()
    execution._check_dependencies(versions)
    versions["torch"] = "different"
    with pytest.raises(ValueError, match="uv.lock"):
        execution._check_dependencies(versions)


def test_cv_wandb_offline_tracks_aggregate_metrics_without_participants(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    bundle, _ = prepared
    condition = next(x for x in verify_bundle(bundle)["conditions"]["cv"]
                     if x["mode"] == "fine_tuning" and x["fold"] == 1)
    output = execution.run_condition(bundle, condition,
        {"output_root": str(tmp_path / "runs"), "wandb": {"mode": "offline"}}, "attempt")
    tracking = json.loads((output / "wandb_run.json").read_text())
    assert tracking["status"] == "finished" and tracking["run_id"]
    research = output / "wandb_research"
    settings = json.loads((research / "settings.json").read_text())
    results = json.loads((research / "results.json").read_text())
    assert settings["study"]["fold"] == 1
    assert "macro_f1" in results["cv"] and "total_seconds" in results["runtime"]
    assert "participant_id" not in "\n".join(path.read_text() for path in research.iterdir())
