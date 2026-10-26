from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from daic_foundation_tab.data.participants import fingerprint

CURRENT_RUN = ContextVar("daic_current_run", default=None)


class RunRecord:
    def __init__(self, artifacts, config):
        self.artifacts = artifacts
        self.started = time.perf_counter()
        self.data = {
            "version": 1,
            "status": "preparing",
            "phase": "preparing",
            "started_at": datetime.now(UTC).isoformat(),
            "dataset_id": config["data"].get("dataset_id"),
            "task": config["experiment"]["task"],
            "input_kind": config.get("input", {}).get("kind", "tabular"),
            "model_name": config["model"]["name"],
            "mode": config["experiment"].get("mode", "fine_tuning"),
            "paired_comparison_verified": False,
        }
        self.update()

    def update(self, **values):
        self.data.update(values)
        self.data.update(
            updated_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - self.started,
        )
        path = self.artifacts.path / "study_run.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(path)

    @contextmanager
    def lifecycle(self):
        token = CURRENT_RUN.set(self)
        try:
            yield self
        except BaseException as error:
            self.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        else:
            self.update(status="completed", phase="completed")
        finally:
            CURRENT_RUN.reset(token)


def record_dataset(artifacts, config, dataset):
    record = CURRENT_RUN.get()
    if record is None:
        return
    manifest = getattr(dataset, "partition_manifest", None)
    if manifest is None:
        manifest = getattr(dataset, "manifest", None)
        if not isinstance(manifest, dict):
            manifest = None
    values = {"dataset_cache_key": dataset.cache_key}
    if manifest:
        artifacts.json("participant_manifest.json", manifest)
        ids = [
            (str(row["participant_id"]), row["split"], row["role"])
            for row in manifest["participants"]
            if row["role"] != "unused"
        ]
        values.update(
            manifest_id=manifest["manifest_id"],
            cohort_id=fingerprint(ids),
            label_budget=manifest["label_budget"],
            fit_labels=sum(row["role"] == "fit" for row in manifest["participants"]),
            selection_labels=sum(row["role"] == "selection" for row in manifest["participants"]),
            selection_labels_used=config["experiment"].get("mode", "fine_tuning") == "fine_tuning",
            paired_comparison_verified=True,
        )
    record.update(**values)


def phase_status(name: str):
    record = CURRENT_RUN.get()
    if record is None:
        return
    status = (
        "training"
        if "training" in name
        else "predicting"
        if "predict" in name
        else "evaluating"
        if any(word in name for word in ("evaluat", "bootstrap", "holdout"))
        else "preparing"
    )
    record.update(status=status, phase=name)


def completed_run(path: Path) -> bool:
    record = path / "study_run.json"
    return not record.is_file() or json.loads(record.read_text())["status"] == "completed"
