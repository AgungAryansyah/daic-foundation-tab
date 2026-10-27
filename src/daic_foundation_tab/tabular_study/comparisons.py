from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab.tabular_study.collection import GROUP_KEYS, _metrics, _scalars
from daic_foundation_tab.tracking.logging import log_progress, phase


def _aligned(
    left: pd.DataFrame, right: pd.DataFrame, task: str, *, folds: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [table.sort_values("participant_id").reset_index(drop=True) for table in (left, right)]
    if any(table.empty or table["participant_id"].duplicated().any() for table in tables):
        raise ValueError("Paired predictions require unique participants")
    if not tables[0]["participant_id"].equals(tables[1]["participant_id"]):
        raise ValueError("Paired prediction participants differ")
    if folds and not tables[0]["fold"].equals(tables[1]["fold"]):
        raise ValueError("Paired predictions use different folds")
    target = "y_true" if task == "classification" else "phq8_true"
    if tables[0][target].isna().any() or not np.array_equal(
        tables[0][target].to_numpy(dtype=float), tables[1][target].to_numpy(dtype=float)
    ):
        raise ValueError("Paired prediction targets differ or are missing")
    return tables[0], tables[1]


def paired_bootstrap(
    left: pd.DataFrame, right: pd.DataFrame, task: str, *, iterations: int, seed: int, progress=None
) -> list[dict]:
    left, right = _aligned(left, right, task, folds=True)
    estimates = [_scalars(_metrics(task, table)) for table in (left, right)]
    names = sorted(set(estimates[0]) & set(estimates[1]))
    distribution = {name: [] for name in names}
    groups = [group.index.to_numpy() for _, group in left.groupby("fold", sort=True)]
    rng = np.random.default_rng(seed)
    if progress:
        progress(0, iterations)
    for completed in range(1, iterations + 1):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        sampled = [_scalars(_metrics(task, table.iloc[indices])) for table in (left, right)]
        for name in names:
            if name in sampled[0] and name in sampled[1]:
                distribution[name].append(sampled[1][name] - sampled[0][name])
        if progress and (completed % max(1, iterations // 20) == 0 or completed == iterations):
            progress(completed, iterations)
    rows = []
    for name in names:
        values = distribution[name]
        rows.append(
            {
                "metric": name,
                "frozen_icl": estimates[0][name],
                "fine_tuning": estimates[1][name],
                "delta_ft_minus_icl": estimates[1][name] - estimates[0][name],
                "ci_lower": float(np.quantile(values, 0.025)) if values else None,
                "ci_upper": float(np.quantile(values, 0.975)) if values else None,
                "valid_bootstrap_samples": len(values),
                "iterations": iterations,
            }
        )
    return rows


def _fold_pairs(results: dict) -> list[dict]:
    grouped = defaultdict(dict)
    for value in results.values():
        condition = value["result"]["condition"]
        key = (
            condition["family"],
            condition["modality"],
            condition["task"],
            condition["model_seed"],
            condition["fold"],
        )
        grouped[key][condition["mode"]] = value
    rows = []
    for key, methods in sorted(grouped.items()):
        if set(methods) != {"frozen_icl", "fine_tuning"}:
            continue
        left = methods["frozen_icl"]["path"]
        right = methods["fine_tuning"]["path"]
        records = [json.loads((path / "study_run.json").read_text()) for path in (left, right)]
        if (
            records[0]["manifest_id"] != records[1]["manifest_id"]
            or records[0]["cohort_id"] != records[1]["cohort_id"]
        ):
            raise ValueError("Paired fold provenance differs")
        tables = [
            pd.read_csv(path / "predictions_cv.csv", dtype={"participant_id": str})
            for path in (left, right)
        ]
        tables = _aligned(*tables, key[2], folds=False)
        metrics = [_scalars(_metrics(key[2], table)) for table in tables]
        identity = dict(zip(("family", "modality", "task", "model_seed", "fold"), key, strict=True))
        for name in sorted(set(metrics[0]) & set(metrics[1])):
            rows.append(
                {
                    **identity,
                    "metric": name,
                    "frozen_icl": metrics[0][name],
                    "fine_tuning": metrics[1][name],
                    "delta_ft_minus_icl": metrics[1][name] - metrics[0][name],
                }
            )
    return rows


def write_cv_comparisons(study: dict, results: dict, oof: dict, destination: Path, logger) -> None:
    pd.DataFrame(_fold_pairs(results)).to_csv(destination / "cv_fold_differences.csv", index=False)
    paired = []
    settings = study["configuration"]["tabular_study"]
    for key in sorted(oof):
        identity = dict(zip(GROUP_KEYS, key, strict=True))
        if identity["mode"] != "frozen_icl":
            continue
        right_key = (key[0], key[1], key[2], "fine_tuning", key[4])
        if right_key not in oof:
            continue
        name = f"bootstrapping {key[0]} {key[1]} {key[2]} seed {key[4]}"
        with phase(logger, name) as started:
            rows = paired_bootstrap(
                oof[key],
                oof[right_key],
                key[2],
                iterations=settings["bootstrap_iterations"],
                seed=settings["bootstrap_seed"],
                progress=lambda n, total, name=name: log_progress(logger, name, n, total, started),
            )
        identity.pop("mode")
        paired.extend({**identity, **row} for row in rows)
    pd.DataFrame(paired).to_csv(destination / "cv_paired_oof.csv", index=False)


def write_final_comparisons(results: dict, destination: Path) -> None:
    grouped = defaultdict(dict)
    for value in results.values():
        condition = value["result"]["condition"]
        key = (
            condition["family"],
            condition["modality"],
            condition["task"],
            condition["model_seed"],
        )
        grouped[key][condition["mode"]] = value
    rows = []
    for key, methods in sorted(grouped.items()):
        if set(methods) != {"frozen_icl", "fine_tuning"}:
            continue
        for split in ("dev", "test"):
            paths = [methods[mode]["path"] for mode in ("frozen_icl", "fine_tuning")]
            if not all((path / f"metrics_{split}.json").is_file() for path in paths):
                continue
            tables = [
                pd.read_csv(path / f"predictions_{split}.csv", dtype={"participant_id": str})
                for path in paths
            ]
            tables = _aligned(*tables, key[2], folds=False)
            values = [_scalars(_metrics(key[2], table)) for table in tables]
            identity = dict(zip(("family", "modality", "task", "model_seed"), key, strict=True))
            for name in sorted(set(values[0]) & set(values[1])):
                rows.append(
                    {
                        **identity,
                        "split": split,
                        "metric": name,
                        "frozen_icl": values[0][name],
                        "fine_tuning": values[1][name],
                        "delta_ft_minus_icl": values[1][name] - values[0][name],
                    }
                )
    pd.DataFrame(rows).to_csv(destination / "final_differences.csv", index=False)
