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
