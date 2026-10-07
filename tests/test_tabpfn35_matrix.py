from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daic_foundation_tab import matrix
from daic_foundation_tab.config import load_config, write_config
from daic_foundation_tab.tracking.artifacts import RunArtifacts


def _fake_execution(monkeypatch, tmp_path, fail_kind=None):
    output_root = tmp_path / "outputs"
    calls = []
    failure = [fail_kind] if fail_kind else []

    def config(path):
        value = load_config(path)
        value["project"]["output_root"] = str(output_root)
        return value

    def launch(preset, progress_callback):
        calls.append(preset)
        output = output_root / f"attempt_{len(calls)}"
        output.mkdir(parents=True)
        value = config(preset)
        write_config(value, output / "config_resolved.yaml")
        (output / "metrics_test.json").write_text('{"mae": 1.0, "roc_auc": 0.8}')
        pd.DataFrame({"metric": [1]}).to_csv(output / "metrics_test.csv", index=False)
        progress_callback()
        kind = "smoke" if preset.stem.endswith("_smoke") else "full"
        if failure and failure[0] == kind:
            failure.pop()
            raise RuntimeError("synthetic failure")
        return output

    def verify(output, value):
        if (output / "corrupt").exists():
            raise ValueError("synthetic corrupt checkpoint")
        assert output.is_dir()

    monkeypatch.setattr(matrix, "load_config", config)
    monkeypatch.setattr(matrix, "_launch", launch)
    monkeypatch.setattr(matrix, "verify_run", verify)
    monkeypatch.setattr(matrix, "require_cuda_device", lambda device: device)
    return calls, output_root / "tabpfn35_matrix_status.json"


def test_matrix_runs_smokes_first_and_retries_only_unverified_outputs(monkeypatch, tmp_path):
    calls, status = _fake_execution(monkeypatch, tmp_path)
    matrix.run_matrix(status_path=status)
    assert len(calls) == 16
    assert all(path.stem.endswith("_smoke") for path in calls[:4])
    assert all(not path.stem.endswith("_smoke") for path in calls[4:])
    state = json.loads(status.read_text())
    assert all(attempts[-1]["status"] == "complete" for attempts in state["presets"].values())
    matrix.run_matrix(status_path=status)
    assert len(calls) == 16
    first = state["presets"][calls[4].name][-1]
    old_output = Path(first["output_directory"])
    (old_output / "corrupt").touch()
    matrix.run_matrix(status_path=status)
    assert len(calls) == 17 and calls[-1] == calls[4]
    assert old_output.is_dir()
    attempts = json.loads(status.read_text())["presets"][calls[-1].name]
    assert len(attempts) == 2 and attempts[-1]["output_directory"] != str(old_output)
    for name in ("test_evaluations.csv", "regression_test_evaluations.csv"):
        assert len(pd.read_csv(status.parent / name)) >= 8
    matrix.run_matrix(status_path=status, verify_only=True)
    assert len(calls) == 17
    state = json.loads(status.read_text())
    state["presets"][calls[0].name][-1]["status"] = "running"
    status.write_text(json.dumps(state))
    matrix.run_matrix(status_path=status)
    interrupted = json.loads(status.read_text())["presets"][calls[0].name]
    assert interrupted[-2]["status"] == "interrupted"
    assert len(calls) == 18


@pytest.mark.parametrize("kind,initial_count", [("smoke", 4), ("full", 16)])
def test_matrix_persists_failure_and_retries_in_a_fresh_directory(
    monkeypatch, tmp_path, kind, initial_count
):
    calls, status = _fake_execution(monkeypatch, tmp_path, fail_kind=kind)
    with pytest.raises(RuntimeError, match="Smoke verification|Matrix incomplete"):
        matrix.run_matrix(status_path=status)
    assert len(calls) == initial_count
    failures = [
        attempts[-1]
        for attempts in json.loads(status.read_text())["presets"].values()
        if attempts[-1]["status"] == "failed"
    ]
    assert len(failures) == 1
    assert "synthetic failure" in failures[0]["failure"]
    assert failures[0]["elapsed_seconds"] >= 0
    assert Path(failures[0]["output_directory"]).is_dir()
    matrix.run_matrix(status_path=status)
    assert len(calls) == 17
    attempts = json.loads(status.read_text())["presets"][failures[0]["preset"]]
    assert attempts[-1]["status"] == "complete"
    assert attempts[-1]["output_directory"] != failures[0]["output_directory"]


def test_run_artifact_directories_are_distinct_for_immediate_retries(tmp_path):
    first = RunArtifacts(tmp_path, "tabpfn35_ft", "audio_visual", 42)
    second = RunArtifacts(tmp_path, "tabpfn35_ft", "audio_visual", 42)
    assert first.path != second.path


def test_launcher_updates_status_while_child_process_is_silent(monkeypatch, tmp_path):
    polls = []

    class Process:
        returncode = 0

        def __init__(self, args, **kwargs):
            self.args = args
            assert args[-2:] == ["--config", "preset.yaml"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, timeout):
            assert timeout == 30
            if not polls:
                raise matrix.subprocess.TimeoutExpired(self.args, timeout)
            return f"{tmp_path}\n", None

    monkeypatch.setattr(matrix.subprocess, "Popen", Process)
    assert matrix._launch(Path("preset.yaml"), lambda: polls.append("heartbeat")) == tmp_path
    assert polls == ["heartbeat"]
