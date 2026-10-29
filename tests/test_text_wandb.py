import json
import sys
from types import SimpleNamespace

from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.text_wandb import TextTracker
from tests.test_text_config import text_config


def test_text_tracking_uses_an_allowlist_without_transcripts_or_participant_ids(
    tmp_path, monkeypatch
):
    captured = {}
    logs = []
    run = SimpleNamespace(
        id="test-run", url="local", log=logs.append, summary={},
        define_metric=lambda *args, **kwargs: None, finish=lambda **kwargs: None,
    )

    def init(**kwargs):
        captured.update(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    )
    config = text_config()
    config["tracking"]["wandb"].update(enabled=True, mode="online")
    config["tracking"]["wandb"]["group"] = "matched-classification"
    config["study"] = {"campaign_id": "matched", "manifest_id": "hash", "private": "PRIVATE"}
    config["data"]["transcripts"]["text"] = "PRIVATE TRANSCRIPT"
    config["data"]["participant_id"] = "PRIVATE ID"
    artifacts = RunArtifacts(tmp_path, "model", "transcript", 1)
    tracker = TextTracker(
        config, artifacts, {"model_id": "Qwen/Qwen3.5-2B", "checkpoint_revision": "fixture"}
    )
    tracker.metrics("dev", {"mae": 1.0, "invalid": float("nan"), "raw_answer": "PRIVATE ANSWER"})
    tracker.metrics("runtime", {"fit_seconds": 2.0, "peak_cuda_allocated_mb": 32.0})
    tracker.finish()
    assert "PRIVATE" not in str(captured)
    assert captured["settings"]["save_code"] is False
    assert captured["group"] == "matched-classification"
    assert captured["config"]["study"] == {"campaign_id": "matched", "manifest_id": "hash"}
    assert logs == [
        {"dev/mae": 1.0},
        {"runtime/fit_seconds": 2.0, "runtime/peak_cuda_allocated_mb": 32.0},
    ]
    assert run.summary["runtime/fit_seconds"] == 2.0


def test_transcript_campaign_tracks_fold_provenance_and_disables_console(tmp_path, monkeypatch):
    captured = {}
    run = SimpleNamespace(id="campaign", url="local", log=lambda values: None, summary={},
                          define_metric=lambda *args, **kwargs: None, finish=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: (captured.update(kwargs), run)[1], Settings=lambda **kwargs: kwargs))
    config = text_config()
    config["tracking"]["wandb"].update(enabled=True, mode="online")
    config["study"] = {"transcript_study_id": "study", "parent_study_id": "parent", "condition_id": "condition",
                       "fold": 1, "stage": "cv", "cpu_threads": 1, "participant_id": "PRIVATE"}
    tracker = TextTracker(config, RunArtifacts(tmp_path, "qwen", "transcript", 42),
                          {"model_id": "Qwen/Qwen3.5-2B", "checkpoint_revision": "pin"},
                          environment={"git_commit": "commit", "cpu_threads": 1,
                                       "container_ram_limit_mb": 16384, "private_path": "PRIVATE"})
    tracker.finish()
    assert captured["settings"]["console"] == "off"
    assert captured["config"]["dataset_id"] == "synthetic"
    assert captured["config"]["study"]["fold"] == 1
    assert captured["config"]["environment"]["cpu_threads"] == 1
    assert "PRIVATE" not in str(captured)


def test_native_offline_campaign_tracking_finishes_with_aggregate_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    config = text_config()
    config["tracking"]["wandb"].update(enabled=True, mode="offline")
    config["study"] = {"transcript_study_id": "study", "parent_study_id": "parent", "fold": 1}
    artifacts = RunArtifacts(tmp_path, "qwen35_2b", "transcript", 42)
    tracker = TextTracker(config, artifacts, {"model_id": "Qwen/Qwen3.5-2B", "checkpoint_revision": "pinned"})
    tracker.metrics("cv", {"macro_f1": .5, "raw_answer": "PRIVATE"})
    tracker.metrics("runtime", {"peak_process_ram_mb": 512, "participants_per_second": 2})
    assert tracker.run.config["study"]["fold"] == 1
    assert tracker.run.summary["cv/macro_f1"] == .5
    tracker.finish()
    record = json.loads((artifacts.path / "wandb_run.json").read_text())
    assert record["status"] == "completed" and record["run_id"]
