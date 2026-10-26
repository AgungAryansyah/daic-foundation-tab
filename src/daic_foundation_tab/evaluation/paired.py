from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .regression import regression_metrics


def paired_comparison(
    left: Path, right: Path, *, split="dev", iterations=1000, seed=42, progress=None
):
    if split not in {"dev", "test"} or iterations < 1:
        raise ValueError("Use dev/test and a positive bootstrap iteration count")
    runs = [Path(left), Path(right)]
    records = []
    for run in runs:
        path = run / "study_run.json"
        if not path.is_file():
            raise ValueError("Legacy run has no verified participant provenance")
        record = json.loads(path.read_text())
        if record["status"] != "completed" or not record.get("paired_comparison_verified"):
            raise ValueError(
                "Paired comparison requires completed runs with verified participant manifests"
            )
        records.append(record)
    for key in ("dataset_id", "task", "manifest_id", "cohort_id", "label_budget"):
        if records[0].get(key) is None or records[0][key] != records[1].get(key):
            raise ValueError(f"Incompatible paired comparison: {key}")
    tables = [
        pd.read_csv(run / f"predictions_{split}.csv", dtype={"participant_id": str})
        .sort_values("participant_id")
        .reset_index(drop=True)
        for run in runs
    ]
    for table in tables:
        if table.empty or table["participant_id"].duplicated().any():
            raise ValueError("Predictions must contain unique participants")
        if set(table["split"]) != {split}:
            raise ValueError("Prediction split mismatch")
    if not tables[0]["participant_id"].equals(tables[1]["participant_id"]):
        raise ValueError("Prediction participant sets differ")
    task = records[0]["task"]
    target_column = "y_true" if task == "classification" else "phq8_true"
    if tables[0][target_column].isna().any() or not tables[0][target_column].equals(
        tables[1][target_column]
    ):
        raise ValueError("Paired comparison requires identical observed targets")
    columns = (
        [target_column, "y_pred", "prob_depressed"]
        if task == "classification"
        else [target_column, "phq8_pred"]
    )
    if any(not np.isfinite(table[columns].to_numpy(dtype=float)).all() for table in tables):
        raise ValueError("Predictions and targets must be finite")

    def metrics(table):
        if task == "classification":
            return classification_metrics(table["y_true"], table["y_pred"], table["prob_depressed"])
        return regression_metrics(table["phq8_true"], table["phq8_pred"])

    estimates = [metrics(table) for table in tables]
    names = [name for name, value in estimates[0].items() if isinstance(value, float)]
    distribution = {name: [] for name in names}
    generator = np.random.default_rng(seed)
    if progress:
        progress(0, iterations)
    for completed in range(1, iterations + 1):
        indices = generator.integers(0, len(tables[0]), size=len(tables[0]))
        sampled = [metrics(table.iloc[indices]) for table in tables]
        for name in names:
            delta = sampled[1][name] - sampled[0][name]
            if np.isfinite(delta):
                distribution[name].append(delta)
        if progress and (completed % max(1, iterations // 20) == 0 or completed == iterations):
            progress(completed, iterations)
    rows = []
    for name in names:
        values = distribution[name]
        rows.append(
            {
                "metric": name,
                "left": estimates[0][name],
                "right": estimates[1][name],
                "delta_right_minus_left": estimates[1][name] - estimates[0][name],
                "ci_lower": float(np.quantile(values, 0.025)) if values else None,
                "ci_upper": float(np.quantile(values, 0.975)) if values else None,
                "valid_bootstrap_samples": len(values),
                "iterations": iterations,
            }
        )
    return pd.DataFrame(rows)
