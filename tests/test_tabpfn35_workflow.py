from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from daic_foundation_tab import regression_runner, runner
from daic_foundation_tab.config import load_config, validate_config
from daic_foundation_tab.matrix import verify_run

from .helpers import write_synthetic_dataset
from .test_tabpfn35_adapter import _install_fine_tuner
from .test_wandb_tracking import _FakeWandb


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tabpfn_workflow_reuses_tracking_outputs_and_collectors(monkeypatch, tmp_path, task):
    root = tmp_path / "data"
    config = write_synthetic_dataset(root)
    labels = pd.read_csv(root / "original_labels/train.csv")
    extra = pd.DataFrame(
        {"Participant_ID": range(306, 320), "PHQ8_Binary": [0, 1] * 7, "PHQ8_Score": [2, 11] * 7}
    )
    pd.concat([labels, extra]).to_csv(root / "original_labels/train.csv", index=False)
    for participant in extra["Participant_ID"]:
        for source in (root / "data").glob("300_*"):
            shutil.copyfile(source, source.with_name(source.name.replace("300", str(participant))))
    suffix = "regression_" if task == "regression" else ""
    preset = Path("configs/experiments") / f"tabpfn35_ft_{suffix}audio_visual_complete_cohort.yaml"
    config["model"] = load_config(preset)["model"]
    config["model"]["parameters"]["epochs"] = 2
    config["experiment"]["task"] = task
    config["_config_path"] = str(preset)
    config["evaluation"]["threshold"] = None if task == "regression" else 0.5
    config["evaluation"]["test_predictions"] = True
    config["evaluation"]["repeated_holdout"] = {
        "enabled": True, "repeats": 2, "validation_fraction": 0.2, "seed_start": 0,
    }
    config["bootstrap"] = {"enabled": True, "iterations": 10, "confidence": 0.95, "random_state": 42}
    config["tracking"]["wandb"]["mode"] = "online"
    config["data"]["test_ground_truth"] = "original_labels/full_test_split.csv"
    pd.DataFrame({"Participant_ID": [500, 501], "PHQ_Score": [3, 12]}).to_csv(
        root / config["data"]["test_ground_truth"], index=False
    )
    validate_config(config)
    instances, exports = _install_fine_tuner(
        monkeypatch, [3.0, 2.0, 2.5] if task == "regression" else [0.75, 0.8, 0.76]
    )
    fake = _FakeWandb()
    starts = []
    original_init = fake.init

    def init(**kwargs):
        starts.append(kwargs)
        return original_init(**kwargs)

    monkeypatch.setattr(fake, "init", init)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for module in (runner, regression_runner):
        monkeypatch.setattr(module, "require_cuda_device", lambda device: device)
        monkeypatch.setattr(module, "reset_cuda_peak_memory", lambda device: None)
        monkeypatch.setattr(module, "cuda_peak_memory", lambda device: {"peak_cuda_allocated_mb": 1.0})
        monkeypatch.setattr(module, "environment_metadata", lambda device: {"cuda_device": device})

    output = runner.run_experiment(config)
    verify_run(output, config)

    def saved(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    assert len(starts) == 1
    assert fake.run.finished == [0]
    assert len(instances) == len(exports) == 3
    assert [item.parameters["random_state"] for item in instances] == [42, 0, 1]
    assert len(instances[0].fit_arguments[0]) == 16
    assert len(instances[0].fit_arguments[2]) == 4
    metric = "mse" if task == "regression" else "roc_auc"
    assert saved("finetune_metadata.json")["selection_metric"] == metric
    assert saved("finetune_metadata.json")["selected_epoch"] == 1
    assert saved("environment.json")["random_seeds"]["tabpfn"] == 42
    assert saved("model.json")["model_version"] == "v3.5"
    assert saved("wandb_run.json")["status"] == "finished"
    assert "TabPFN-3.5-FT" in (output / "summary.md").read_text()
    assert "TabICL" not in (output / "summary.md").read_text()
    assert len(pd.read_csv(output / "bootstrap_dev.csv")) == 10
    assert len(pd.read_csv(output / "repeated_holdout_metrics.csv")) == 2
    assert len(list((output / "repeated_splits").glob("*.csv"))) == 2
    assert len(pd.read_csv(output / "predictions_test.csv")) == 2
    aggregate = "regression_test_evaluations.csv" if task == "regression" else "test_evaluations.csv"
    assert pd.read_csv(output.parent / aggregate)["model"].tolist() == [config["model"]["name"]]
    research = saved("wandb_research/results.json")
    assert research["fine_tuning"]["selection_metric"] == metric
    assert len(research["fine_tuning_history"]) == 2
    assert f"val/{metric}" in research["fine_tuning_history"][0]
    uploaded = "\n".join(Path(path).read_text() for path, _ in fake.run.artifacts[0].files)
    for private in (str(tmp_path), "checkpoint_path", "participant_id", "predictions_test"):
        assert private not in uploaded
    log = (output / "run.log").read_text()
    for phase in ("preparing features", "training", "evaluating development", "evaluating test"):
        assert f"phase={phase} status=complete" in log
    assert "progress=2/2 repeats (100.0%)" in log
    assert "progress=10/10 resamples (100.0%)" in log
    checkpoint = Path(saved("finetune_metadata.json")["checkpoint_path"])
    checkpoint.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="hash"):
        verify_run(output, config)
