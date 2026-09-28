from __future__ import annotations

import csv
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

_IDENTITY_COLUMNS = ("run", "experiment", "dataset", "model", "feature_set", "seed", "threshold")
_REGRESSION_IDENTITY_COLUMNS = _IDENTITY_COLUMNS[:-1]


def _run_config(run: Path) -> dict[str, Any]:
    return yaml.safe_load((run / "config_resolved.yaml").read_text(encoding="utf-8"))


def _evaluation_row(metrics_path: Path) -> dict[str, Any]:
    run = metrics_path.parent
    config = _run_config(run)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold_path = run / "decision_threshold.json"
    threshold = (
        json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
        if threshold_path.is_file()
        else 0.5
    )
    return {
        "run": run.name,
        "experiment": config["experiment"].get("name", ""),
        "dataset": Path(config["data"]["root"]).name,
        "model": config["model"]["name"],
        "feature_set": config["experiment"]["feature_set"],
        "seed": config["project"]["seed"],
        "threshold": threshold,
        **{
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
    }


def collect_test_evaluations(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "test_evaluations.csv"
    with (root / ".test_evaluations.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [
            _evaluation_row(path.with_name("metrics_test.json"))
            for path in sorted(root.glob("*/metrics_test.csv"))
            if _run_config(path.parent)["experiment"].get("task", "classification")
            == "classification"
        ]
        metric_columns = sorted({key for row in rows for key in row} - set(_IDENTITY_COLUMNS))
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=root, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=[*_IDENTITY_COLUMNS, *metric_columns])
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def collect_regression_test_evaluations(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "regression_test_evaluations.csv"
    with (root / ".regression_test_evaluations.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for path in sorted(root.glob("*/metrics_test.csv")):
            run = path.parent
            config = _run_config(run)
            if config["experiment"].get("task") != "regression":
                continue
            metrics = json.loads((run / "metrics_test.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "run": run.name,
                    "experiment": config["experiment"].get("name", ""),
                    "dataset": Path(config["data"]["root"]).name,
                    "model": config["model"]["name"],
                    "feature_set": config["experiment"]["feature_set"],
                    "seed": config["project"]["seed"],
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                    },
                }
            )
        metric_columns = sorted(
            {key for row in rows for key in row} - set(_REGRESSION_IDENTITY_COLUMNS)
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=root, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(
                stream, fieldnames=[*_REGRESSION_IDENTITY_COLUMNS, *metric_columns]
            )
            writer.writeheader()
            writer.writerows(rows)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination
