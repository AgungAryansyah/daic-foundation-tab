from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daic_foundation_tab import matrix

from .test_tabpfn35_matrix import _fake_execution


def test_kumo_matrix_runs_two_smokes_and_six_full_presets_then_resumes(monkeypatch, tmp_path):
    calls, status = _fake_execution(monkeypatch, tmp_path)
    matrix.run_matrix(status_path=status, model_family="kumo_medium")
    assert len(calls) == 8
    assert all(path.stem.endswith("_smoke") for path in calls[:2])
    assert all(not path.stem.endswith("_smoke") for path in calls[2:])
    assert all("edaic" in path.name for path in calls)
    matrix.run_matrix(status_path=status, model_family="kumo_medium", verify_only=True)
    assert len(calls) == 8
    for name in ("test_evaluations.csv", "regression_test_evaluations.csv"):
        assert len(pd.read_csv(status.parent / name)) == 4
    state = json.loads(status.read_text())
    output = Path(state["presets"][calls[2].name][-1]["output_directory"])
    (output / "corrupt").touch()
    matrix.run_matrix(status_path=status, model_family="kumo_medium")
    assert len(calls) == 9 and calls[-1] == calls[2]
    assert output.is_dir()


@pytest.mark.parametrize("kind,count", [("smoke", 2), ("full", 8)])
def test_kumo_matrix_smoke_gate_and_fresh_retry(monkeypatch, tmp_path, kind, count):
    calls, status = _fake_execution(monkeypatch, tmp_path, fail_kind=kind)
    with pytest.raises(RuntimeError, match="Smoke verification|Matrix incomplete"):
        matrix.run_matrix(status_path=status, model_family="kumo_medium")
    assert len(calls) == count
    matrix.run_matrix(status_path=status, model_family="kumo_medium")
    assert len(calls) == 9
    attempts = [value for value in json.loads(status.read_text())["presets"].values() if len(value) == 2]
    assert len(attempts) == 1
    assert attempts[0][0]["status"] == "failed"
    assert attempts[0][1]["status"] == "complete"
    assert attempts[0][0]["output_directory"] != attempts[0][1]["output_directory"]


def test_kumo_matrix_uses_its_own_default_status_file(monkeypatch, tmp_path):
    calls, _ = _fake_execution(monkeypatch, tmp_path)
    configs = Path("configs/experiments").resolve()
    monkeypatch.chdir(tmp_path)
    matrix.run_matrix(config_directory=configs, model_family="kumo_medium")
    assert len(calls) == 8
    assert Path("outputs/kumo_medium_matrix_status.json").is_file()
    assert not Path("outputs/tabpfn35_matrix_status.json").exists()
