from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from daic_foundation_tab.tracking.artifacts import RunArtifacts
from daic_foundation_tab.tracking.wandb import WandbTracker, sanitized_wandb_config


class _FakeArtifact:
    def __init__(self, name, type, description, metadata) -> None:
        self.name = name
        self.type = type
        self.description = description
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name) -> None:
        self.files.append((path, name))


class _FakeRun:
    def __init__(self) -> None:
        self.id = "run-123"
        self.url = "https://wandb.example/run-123"
        self.logs = []
        self.artifacts = []
        self.finished = []

    def log(self, values) -> None:
        self.logs.append(values)

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self, exit_code) -> None:
        self.finished.append(exit_code)


class _FakeWandb:
    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_arguments = None

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_arguments = kwargs
        return self.run

    Artifact = _FakeArtifact

    class Table:
        def __init__(self, columns, data) -> None:
            self.columns = columns
            self.data = data


def _config(mode: str = "online") -> dict:
    return {
        "project": {"seed": 42, "output_root": "/private/outputs", "cache_root": "/private/cache"},
        "data": {
            "root": "/private/data",
            "labels": {"train": "/private/train.csv"},
            "missing_modality_policy": "exclude",
            "max_exclusion_fraction": 0.05,
            "source_groups": {
                "covarep": {
                    "modality": "audio",
                    "parser": "covarep",
                    "pattern": "/private/{participant_id}.csv",
                }
            },
        },
        "modalities": {"audio": {"enabled": True, "groups": ["covarep"]}},
        "aggregation": {"statistics": ["mean", "std"]},
        "experiment": {
            "name": "fine_tuning_study",
            "task": "classification",
            "feature_set": "audio",
        },
        "model": {
            "name": "tabiclv2_ft",
            "checkpoint_version": "tabicl-classifier-v2",
            "parameters": {"epochs": 50, "eval_metric": "roc_auc"},
        },
        "evaluation": {
            "test_predictions": False,
            "fine_tuning": {"validation_fraction": 0.2},
        },
        "bootstrap": {"enabled": True},
        "tracking": {
            "wandb": {
                "enabled": True,
                "project": "daic-foundation-tab",
                "entity": None,
                "mode": mode,
                "tags": ["research"],
            }
        },
    }


def _validation() -> dict:
    return {
        "feature_set": "audio",
        "feature_count_before_filtering": 3,
        "dropped_all_nan": ["feature_a"],
        "dropped_constant": [],
        "near_constant": [],
        "split_statistics": {
            "train": {
                "participants": 10,
                "depressed": 4,
                "non_depressed": 6,
                "positive_rate": 0.4,
                "feature_count": 3,
                "missing_cell_percentage": 0.0,
                "complete_cases": 10,
            },
            "dev": {
                "participants": 4,
                "depressed": 2,
                "non_depressed": 2,
                "positive_rate": 0.5,
                "feature_count": 3,
                "missing_cell_percentage": 0.0,
                "complete_cases": 4,
            },
            "test": {
                "participants": 5,
                "depressed": 0,
                "non_depressed": 0,
                "positive_rate": None,
                "feature_count": 3,
                "missing_cell_percentage": 0.0,
                "complete_cases": 5,
            },
        },
        "exclusions": {"dev": {"participants": 5, "excluded": 1, "exclusion_fraction": 0.2}},
    }


def test_sanitized_config_excludes_local_paths() -> None:
    config = sanitized_wandb_config(_config(), "cache-key", "study-123")

    serialized = json.dumps(config)

    assert "/private" not in serialized
    assert "labels" not in config["data"]
    assert "pattern" not in config["data"]["source_groups"]["covarep"]


def test_tracker_logs_only_cohort_level_research_records(monkeypatch, tmp_path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    artifacts = RunArtifacts(tmp_path, "tabiclv2_ft", "audio", 42)
    tracker = WandbTracker.start(_config(), artifacts, _validation(), "cache-key")
    tracker.record_fine_tuning(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1]),
        2,
        12.5,
        {
            "selection_metric": "roc_auc",
            "best_validation_metric": 0.75,
            "checkpoint_path": "/private/best.ckpt",
            "checkpoint_sha256": "abc123",
        },
    )
    tracker.record_development({"macro_f1": 0.6, "confusion_matrix": [[1, 0], [1, 2]]})
    tracker.record_bootstrap({"macro_f1": {"estimate": 0.6, "ci_lower": 0.4, "ci_upper": 0.8}})
    tracker.record_repeated_holdout(
        pd.DataFrame({"seed": [0, 1], "macro_f1": [0.5, 0.6]}),
        {"macro_f1": {"mean": 0.55, "std": 0.05}},
    )
    tracker.complete(
        {"model_name": "TabICLv2-FT", "checkpoint": "tabicl-classifier-v2"},
        {"python": "3.13", "gpu_name": "GPU"},
        {"total_seconds": 15.0},
    )

    assert fake_wandb.init_arguments["mode"] == "online"
    assert fake_wandb.init_arguments["resume"] == "never"
    assert fake_wandb.init_arguments["group"].startswith("study-")
    assert fake_wandb.init_arguments["settings"] == {
        "disable_git": True,
        "save_code": False,
        "x_save_requirements": False,
    }
    assert {"tabiclv2_ft", "audio", "classification", "full", "research"} == set(
        fake_wandb.init_arguments["tags"]
    )
    assert fake_wandb.run.finished == [0]
    assert len(fake_wandb.run.artifacts) == 1
    uploaded_names = {name for _, name in fake_wandb.run.artifacts[0].files}
    assert uploaded_names == {"data_card.json", "results.json", "settings.json"}
    artifact_contents = "\n".join(
        Path(path).read_text(encoding="utf-8") for path, _ in fake_wandb.run.artifacts[0].files
    )
    for path, _ in fake_wandb.run.artifacts[0].files:
        assert "/private" not in Path(path).read_text(encoding="utf-8")
    assert "participant_id" not in artifact_contents
    assert "checkpoint_path" not in artifact_contents
    assert "predictions_dev.csv" not in artifact_contents
    record = json.loads((artifacts.path / "wandb_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "finished"
    assert record["artifact_references"] == [
        {"name": fake_wandb.run.artifacts[0].name, "type": "research-record"}
    ]


def test_tracker_loads_wandb_key_from_project_dotenv(monkeypatch, tmp_path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    (tmp_path / ".env").write_text("WANDB_API_KEY=test-key\n", encoding="utf-8")
    artifacts = RunArtifacts(tmp_path / "outputs", "tabiclv2_ft", "audio", 42)

    WandbTracker.start(_config(), artifacts, _validation(), "cache-key")

    assert os.environ["WANDB_API_KEY"] == "test-key"


def test_tracker_supports_offline_mode(monkeypatch, tmp_path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    artifacts = RunArtifacts(tmp_path, "tabiclv2_ft", "audio", 42)

    tracker = WandbTracker.start(_config(mode="offline"), artifacts, _validation(), "cache-key")
    tracker.complete({}, {}, {"total_seconds": 1.0})

    assert fake_wandb.init_arguments["mode"] == "offline"
    assert fake_wandb.init_arguments["dir"] == artifacts.path
    assert fake_wandb.run.finished == [0]


def test_disabled_tracker_does_not_initialize_wandb(tmp_path) -> None:
    config = _config(mode="disabled")
    artifacts = RunArtifacts(tmp_path, "tabiclv2_ft", "audio", 42)

    tracker = WandbTracker.start(config, artifacts, _validation(), "cache-key")
    tracker.fail()

    record = json.loads((artifacts.path / "wandb_run.json").read_text(encoding="utf-8"))
    assert tracker.run is None
    assert record["status"] == "failed"
