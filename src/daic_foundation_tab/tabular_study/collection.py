from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.regression import regression_metrics
from daic_foundation_tab.tabular_study.bundle import _json, file_hash, read_bundle
from daic_foundation_tab.tabular_study.results import verify_result
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

GROUP_KEYS = ("family", "modality", "task", "mode", "model_seed")


def _scalars(values: dict) -> dict:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    }


def _metrics(task: str, table: pd.DataFrame) -> dict:
    if task == "classification":
        return classification_metrics(table["y_true"], table["y_pred"], table["prob_depressed"])
    return regression_metrics(table["phq8_true"], table["phq8_pred"])


def _discover(context: tuple, roots: list[Path], stage: str, logger) -> tuple[dict, dict, list]:
    study = context[0]
    results, attempts, conflicts = {}, {}, []
    expected_ids = {item["condition_id"] for item in study["conditions"][stage]}
    paths = sorted({path.resolve() for root in roots for path in root.rglob("result.json")})
    status_paths = sorted({path.resolve() for root in roots for path in root.rglob("status.json")})
    for path in status_paths:
        status = json.loads(path.read_text())
        if status.get("study_id") == study["study_id"]:
            attempts.update(status.get("conditions", {}))
    with phase(logger, "verifying condition results") as started:
        log_progress(logger, "verifying condition results", 0, len(paths), started)
        versions = None
        source_hashes = {}
        for completed, path in enumerate(paths, 1):
            candidate = json.loads(path.read_text())
            if (
                candidate.get("study_id") == study["study_id"]
                and candidate.get("condition", {}).get("stage") == stage
                and candidate.get("condition", {}).get("condition_id") in expected_ids
            ):
                result = verify_result(context, path.parent)
                identity = (result["condition"]["family"], result["condition"]["task"])
                record = json.loads((path.parent / "study_run.json").read_text())
                source_hash = record["source_checkpoint_sha256"]
                if identity in source_hashes and source_hashes[identity] != source_hash:
                    raise ValueError("Source model checkpoints differ across results")
                source_hashes[identity] = source_hash
                condition_id = result["condition"]["condition_id"]
                digest = file_hash(path)
                if condition_id in results:
                    if results[condition_id]["digest"] != digest:
                        conflicts.append(condition_id)
                else:
                    environment = json.loads((path.parent / "environment.json").read_text())
                    current_versions = environment.get("dependency_versions")
                    if not isinstance(current_versions, dict):
                        raise ValueError("Result has no dependency versions")
                    if versions is not None and versions != current_versions:
                        raise ValueError("Result dependency versions differ across systems")
                    versions = current_versions
                    results[condition_id] = {
                        "result": result,
                        "path": path.parent,
                        "digest": digest,
                        "environment": environment,
                    }
            log_progress(logger, "verifying condition results", completed, len(paths), started)
    return results, attempts, sorted(set(conflicts))


def _coverage(study: dict, stage: str, results: dict, attempts: dict, conflicts: list) -> list:
    rows = []
    for condition in study["conditions"][stage]:
        condition_id = condition["condition_id"]
        status = (
            "conflicting"
            if condition_id in conflicts
            else "completed"
            if condition_id in results
            else attempts.get(condition_id, {}).get("status", "missing")
        )
        rows.append({**condition, "status": status})
    return rows


def _write_condition_metrics(results: dict, destination: Path, stage: str) -> list[dict]:
    rows = []
    for value in results.values():
        condition, path = value["result"]["condition"], value["path"]
        environment = value["environment"]
        row = {
            **condition,
            "output": str(path),
            "host_label": environment.get("host_label"),
            "gpu_name": environment.get("gpu_name"),
            "gpu_vram_bytes": environment.get("total_gpu_vram_bytes"),
        }
        for split in ("cv",) if stage == "cv" else ("dev", "test"):
            metrics_path = path / f"metrics_{split}.json"
            if metrics_path.is_file():
                row.update(
                    {
                        f"{split}/{key}": value
                        for key, value in _scalars(json.loads(metrics_path.read_text())).items()
                    }
                )
        row.update(
            {
                f"runtime/{key}": value
                for key, value in _scalars(json.loads((path / "runtime.json").read_text())).items()
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "condition_metrics.csv", index=False)
    return rows


def _write_cv_summaries(context: tuple, results: dict, destination: Path) -> dict:
    study, table, _, _ = context
    expected_folds = set(range(1, study["configuration"]["tabular_study"]["folds"] + 1))
    expected_ids = set(table.loc[table["split"] == "train", "participant_id"].astype(str))
    groups = defaultdict(list)
    for value in results.values():
        condition = value["result"]["condition"]
        groups[tuple(condition[key] for key in GROUP_KEYS)].append(value)
    predictions_dir = destination / "oof_predictions"
    predictions_dir.mkdir(exist_ok=True)
    summaries, oof = [], {}
    for key, items in sorted(groups.items()):
        if {item["result"]["condition"]["fold"] for item in items} != expected_folds or len(
            items
        ) != len(expected_folds):
            continue
        frames, fold_metrics = [], []
        for item in items:
            fold = item["result"]["condition"]["fold"]
            frame = pd.read_csv(item["path"] / "predictions_cv.csv", dtype={"participant_id": str})
            frame["fold"] = fold
            frames.append(frame)
            fold_metrics.append(
                _scalars(json.loads((item["path"] / "metrics_cv.json").read_text()))
            )
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
        if (
            combined["participant_id"].duplicated().any()
            or set(combined["participant_id"]) != expected_ids
        ):
            raise ValueError("Out-of-fold predictions do not cover each training participant once")
        identity = dict(zip(GROUP_KEYS, key, strict=True))
        path = predictions_dir / f"{fingerprint(identity)[:20]}.csv"
        combined.to_csv(path, index=False)
        oof[key] = combined
        pooled = _scalars(_metrics(identity["task"], combined))
        for metric, pooled_value in pooled.items():
            values = pd.Series(
                [item.get(metric, float("nan")) for item in fold_metrics], dtype=float
            ).dropna()
            summaries.append(
                {
                    **identity,
                    "metric": metric,
                    "fold_mean": values.mean() if len(values) else None,
                    "fold_std": values.std(ddof=1) if len(values) > 1 else None,
                    "valid_folds": len(values),
                    "pooled_oof": pooled_value,
                    "participants": len(combined),
                    "predictions_file": str(path.relative_to(destination)),
                }
            )
    pd.DataFrame(summaries).to_csv(destination / "cv_summary.csv", index=False)
    return oof


def _write_runtime_summary(rows: list[dict], destination: Path) -> None:
    table = pd.DataFrame(rows)
    if table.empty:
        pd.DataFrame().to_csv(destination / "runtime_summary.csv", index=False)
        return
    keys = ["family", "modality", "task", "mode", "gpu_name", "gpu_vram_bytes"]
    metrics = [column for column in table if column.startswith("runtime/")]
    table.groupby(keys, dropna=False)[metrics].agg(["count", "mean", "std"]).to_csv(
        destination / "runtime_summary.csv"
    )


def collect_study(
    bundle: str | Path,
    roots: list[str | Path],
    destination: str | Path,
    *,
    stage: str = "cv",
    allow_partial: bool = False,
) -> Path:
    context = read_bundle(bundle)
    if stage not in context[0]["conditions"]:
        raise ValueError("Stage must be cv or final")
    roots = [Path(path).resolve() for path in roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("Each result root must be an existing directory")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(destination / "collect.log", "INFO")
    results, attempts, conflicts = _discover(context, roots, stage, logger)
    coverage = _coverage(context[0], stage, results, attempts, conflicts)
    missing = [row for row in coverage if row["status"] != "completed"]
    coverage_report = {
            "study_id": context[0]["study_id"],
            "stage": stage,
            "status": "incomplete" if missing else "collecting",
            "expected": len(coverage),
            "completed": len(coverage) - len(missing),
            "conditions": coverage,
        }
    _json(destination / "coverage.json", coverage_report)
    pd.DataFrame(coverage).to_csv(destination / "coverage.csv", index=False)
    if conflicts:
        raise ValueError("Conflicting successful attempts exist for the same condition")
    if missing and not allow_partial:
        raise ValueError(f"Campaign is incomplete: {len(missing)} conditions remain")
    try:
        with phase(logger, "summarizing condition metrics"):
            rows = _write_condition_metrics(results, destination, stage)
            _write_runtime_summary(rows, destination)
            if stage == "cv":
                from daic_foundation_tab.tabular_study.comparisons import write_cv_comparisons

                oof = _write_cv_summaries(context, results, destination)
                write_cv_comparisons(context[0], results, oof, destination, logger)
            else:
                from daic_foundation_tab.tabular_study.comparisons import write_final_comparisons

                write_final_comparisons(results, destination)
    except Exception as error:
        coverage_report.update(status="failed", error_type=type(error).__name__)
        _json(destination / "coverage.json", coverage_report)
        raise
    coverage_report["status"] = "incomplete" if missing else "completed"
    _json(destination / "coverage.json", coverage_report)
    return destination
