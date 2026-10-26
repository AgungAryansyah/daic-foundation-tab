import json

import pandas as pd
import pytest

from daic_foundation_tab import runner, text_runner
from daic_foundation_tab.models.text_base import TextPrediction
from tests.test_text_config import text_config
from tests.test_transcripts import transcript_config


class FakeTextModel:
    def __init__(self, task, fail=False):
        self.task = task
        self.fail = fail
        self.steps = 0

    def prepare(self, demonstrations):
        self.demonstrations = demonstrations

    def preflight(self, examples):
        pass

    def fit(self, training, validation, checkpoint_directory, progress, resume=False):
        assert all(item.target is not None for item in [*training, *validation])
        self.steps = 1
        progress(1, 1)

    def predict(self, examples):
        assert all(item.target is None for item in examples)
        if self.fail:
            raise RuntimeError("prediction failed")
        return [
            TextPrediction(1, 0.75, "1", 50)
            if self.task == "classification"
            else TextPrediction(12.5, None, "12.5", 50)
            for _ in examples
        ]

    def run_metadata(self):
        return {
            "model_id": "Qwen/Qwen3.5-2B",
            "checkpoint_revision": "fixture-revision",
            "adaptation": "lora" if self.steps else "frozen_icl",
            "optimizer_steps": self.steps,
        }


def runner_config(tmp_path, task="classification", mode="frozen_icl"):
    config = transcript_config(tmp_path)
    config["model"] = text_config()["model"]
    config["experiment"].update(task=task, mode=mode)
    config["evaluation"]["test_predictions"] = True
    config["evaluation"]["threshold"] = 0.5 if task == "classification" else None
    config["tracking"]["wandb"]["enabled"] = False
    return config


def mock_gpu(monkeypatch):
    monkeypatch.setattr(text_runner, "require_cuda_device", lambda device: device)
    monkeypatch.setattr(text_runner, "reset_cuda_peak_memory", lambda device: None)
    monkeypatch.setattr(
        text_runner, "cuda_peak_memory", lambda device: {"peak_cuda_allocated_mb": 1}
    )
    monkeypatch.setattr(text_runner, "environment_metadata", lambda device: {"device": device})


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
def test_text_execution_preserves_schema_and_records_completed_run(
    tmp_path, monkeypatch, task, mode
):
    config = runner_config(tmp_path, task, mode)
    mock_gpu(monkeypatch)
    monkeypatch.setattr(text_runner, "create_text_model", lambda config: FakeTextModel(task))
    output = runner.run_experiment(config)
    record = json.loads((output / "study_run.json").read_text())
    assert record["status"] == "completed" and record["paired_comparison_verified"]
    assert record["selection_labels_used"] == (mode == "fine_tuning")
    assert (output / "metrics_dev.json").is_file()
    assert not (output / "metrics_test.json").exists()
    table = pd.read_csv(output / "predictions_test.csv")
    column = "y_true" if task == "classification" else "phq8_true"
    assert table[column].isna().all()
    assert table["participant_id"].tolist() == [500, 501]
    assert "I feel tired" not in (output / "study_run.json").read_text()
    assert "I feel tired" not in (output / "config_resolved.yaml").read_text()


def test_failed_run_can_resume_and_rejects_changed_settings(tmp_path, monkeypatch):
    config = runner_config(tmp_path)
    mock_gpu(monkeypatch)
    monkeypatch.setattr(
        text_runner, "create_text_model", lambda config: FakeTextModel("classification", fail=True)
    )
    with pytest.raises(RuntimeError, match="prediction failed"):
        runner.run_experiment(config)
    output = next((tmp_path / "data/cache/outputs").iterdir())
    assert json.loads((output / "study_run.json").read_text())["status"] == "failed"
    config["_resume_directory"] = str(output)
    config["project"]["seed"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.run_experiment(config)
    config["project"]["seed"] -= 1
    monkeypatch.setattr(
        text_runner, "create_text_model", lambda config: FakeTextModel("classification")
    )
    assert runner.run_experiment(config) == output
    with pytest.raises(ValueError, match="already completed"):
        runner.run_experiment(config)
