import json

import pytest

from daic_foundation_tab import runner
from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.lifecycle import CURRENT_RUN, RunRecord, completed_run
from tests.helpers import write_synthetic_dataset


def test_failures_before_data_preparation_preserve_status(tmp_path, monkeypatch):
    config = write_synthetic_dataset(tmp_path / "data")

    def fail(device):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(runner, "require_cuda_device", fail)
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.run_experiment(config)
    output = next((tmp_path / "data/cache/outputs").iterdir())
    record = json.loads((output / "study_run.json").read_text())
    assert record["status"] == "failed"
    assert not completed_run(output)
    assert CURRENT_RUN.get() is None


def test_success_is_marked_only_after_the_full_lifecycle(tmp_path):
    config = write_synthetic_dataset(tmp_path / "data")
    artifacts = RunArtifacts(tmp_path / "outputs", "model", "audio", 1)
    record = RunRecord(artifacts, config)
    with record.lifecycle():
        assert not completed_run(artifacts.path)
        record.update(phase="evaluating", status="evaluating")
    assert completed_run(artifacts.path)
    assert not record.data["paired_comparison_verified"]
