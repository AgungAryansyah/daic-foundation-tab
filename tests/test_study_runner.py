import fcntl
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from daic_foundation_tab import runner, study_runner
from daic_foundation_tab.config import load_config, write_config
from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.matched_study import prepare_study
from tests.test_matched_study import study_config


def fake_condition(config_path, result_path):
    config = load_config(config_path)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    output = Path(config["project"]["output_root"]) / "fixture-run"
    output.mkdir(parents=True, exist_ok=True)
    write_config(config, output / "config_resolved.yaml")
    record = {
        "status": "completed",
        "paired_comparison_verified": True,
        "dataset_id": config["data"]["dataset_id"],
        "task": config["experiment"]["task"],
        "model_name": config["model"]["name"],
        "mode": config["experiment"]["mode"],
        "input_kind": config["input"]["kind"],
        "manifest_id": config["study"]["manifest_id"],
        "cohort_id": config["study"]["cohort_id"],
        "source_revision": "fixture-revision",
    }
    (output / "study_run.json").write_text(json.dumps(record))
    manifest = json.loads(Path(config["experiment"]["partition_manifest"]).read_text())
    for split in ("dev", "test"):
        pd.DataFrame(
            [
                {
                    "participant_id": row["participant_id"],
                    "split": split,
                    "y_pred": 1,
                    "prob_depressed": 0.75,
                    "phq8_pred": 12.0,
                }
                for row in manifest["participants"]
                if row["split"] == split
            ]
        ).to_csv(output / f"predictions_{split}.csv", index=False)
    is_ft = config["experiment"]["mode"] == "fine_tuning"
    metric = "roc_auc" if config["experiment"]["task"] == "classification" else "mae"
    (output / "metrics_dev.json").write_text(json.dumps({metric: 0.8 if is_ft else 0.6}))
    (output / "runtime.json").write_text(json.dumps({"fit_seconds": 5 if is_ft else 0}))
    (output / "wandb_run.json").write_text(json.dumps({"run_id": "fixture", "url": "local"}))
    result_path.write_text(json.dumps({"identity": fingerprint(payload), "output": str(output)}))


def test_campaign_records_all_conditions_metrics_deltas_and_skips_completed(tmp_path, monkeypatch):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    calls = []

    def execute(config_path, result_path):
        calls.append(config_path)
        fake_condition(config_path, result_path)

    monkeypatch.setattr(runner, "_git_commit", lambda: "fixture-revision")
    monkeypatch.setattr(study_runner, "_execute_condition", execute)
    assert study_runner.run_study(destination) == destination
    assert len(calls) == 16
    state = json.loads((destination / "status.json").read_text())
    assert state["status"] == "completed"
    assert len(state["conditions"]) == 16
    assert all(value["status"] == "completed" for value in state["conditions"].values())
    table = pd.read_csv(destination / "metrics.csv")
    assert len(table) == 16
    differences = pd.read_csv(destination / "adaptation_differences.csv")
    assert len(differences) == 16
    assert differences.loc[
        differences["metric"] == "development/roc_auc", "delta_ft_minus_icl"
    ].tolist() == pytest.approx([0.2] * 4)
    study_runner.run_study(destination)
    assert len(calls) == 16


def test_campaign_failure_restarts_failed_condition_and_preserves_completed(tmp_path, monkeypatch):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    calls = []

    def execute(config_path, result_path):
        calls.append(config_path)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, "worker")
        fake_condition(config_path, result_path)

    monkeypatch.setattr(runner, "_git_commit", lambda: "fixture-revision")
    monkeypatch.setattr(study_runner, "_execute_condition", execute)
    with pytest.raises(subprocess.CalledProcessError):
        study_runner.run_study(destination)
    state = json.loads((destination / "status.json").read_text())
    assert state["status"] == "failed"
    assert [row["status"] for row in state["conditions"].values()] == ["completed", "failed"]
    study_runner.run_study(destination)
    assert len(calls) == 17
    assert calls[0] not in calls[1:]
    assert calls[1] == calls[2]


@pytest.mark.parametrize("change", ["source", "inputs", "completed_run", "prediction_ids"])
def test_campaign_refuses_incompatible_reuse(tmp_path, monkeypatch, change):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    monkeypatch.setattr(runner, "_git_commit", lambda: "fixture-revision")
    monkeypatch.setattr(study_runner, "_execute_condition", fake_condition)
    study_runner.run_study(destination)
    if change == "source":
        monkeypatch.setattr(runner, "_git_commit", lambda: "other-revision")
    elif change == "inputs":
        (tmp_path / "data/data/300_TRANSCRIPT.csv").write_text("changed input")
    else:
        output = next((destination / "runs").glob("*/fixture-run"))
        if change == "completed_run":
            path = output / "study_run.json"
            record = json.loads(path.read_text())
            record["cohort_id"] = "other"
            path.write_text(json.dumps(record))
        else:
            table = pd.read_csv(output / "predictions_dev.csv")
            table.loc[0, "participant_id"] = 999
            table.to_csv(output / "predictions_dev.csv", index=False)
    with pytest.raises(ValueError, match="changed|mismatch|shared participants"):
        study_runner.run_study(destination)


def test_campaign_refuses_concurrent_execution(tmp_path):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    with (destination / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            study_runner.run_study(destination)


def test_worker_runs_selected_config_and_writes_completion_pointer(tmp_path, monkeypatch):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    path = next(destination.glob("*.yaml"))
    result_path = tmp_path / "result.json"
    calls = []

    def run(config):
        calls.append(config)
        return tmp_path / "completed"

    monkeypatch.setattr(runner, "run_experiment", run)
    monkeypatch.setattr(
        sys, "argv", ["worker", "--config", str(path), "--result", str(result_path)]
    )
    study_runner.main()
    assert len(calls) == 1
    assert json.loads(result_path.read_text())["output"] == str(tmp_path / "completed")


def test_worker_command_uses_current_python_and_separate_process(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    study_runner._execute_condition(tmp_path / "condition.yaml", tmp_path / "result.json")
    args, kwargs = calls[0]
    assert args[0][:4] == [sys.executable, "-u", "-m", "daic_foundation_tab.study_runner"]
    assert kwargs["check"] is True


def test_report_failure_keeps_study_incomplete(tmp_path, monkeypatch):
    destination = prepare_study(study_config(tmp_path), tmp_path / "prepared")
    monkeypatch.setattr(runner, "_git_commit", lambda: "fixture-revision")
    monkeypatch.setattr(study_runner, "_execute_condition", fake_condition)

    def fail(*args):
        raise OSError("report failure")

    monkeypatch.setattr(study_runner, "summarize_study", fail)
    with pytest.raises(OSError, match="report failure"):
        study_runner.run_study(destination)
    state = json.loads((destination / "status.json").read_text())
    assert state["status"] == "failed"
    assert state["completed_conditions"] == 16
