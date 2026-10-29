from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from daic_foundation_tab.tabular_study.bundle import _json, file_hash
from daic_foundation_tab.tabular_study.collection import (
    GROUP_KEYS,
    _coverage,
    _write_condition_metrics,
    _write_cv_summaries,
)
from daic_foundation_tab.tabular_study.comparisons import (
    _aligned,
    paired_bootstrap,
    write_cv_comparisons,
    write_final_comparisons,
)
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

from .audit import require_audit
from .bundle import read_bundle
from .results import verify_result


def _discover(context: tuple, roots: list[Path], stage: str, logger) -> tuple[dict, dict]:
    study = context[0]
    results, attempts, dependencies, test_targets = {}, {}, None, {}
    for root in roots:
        for path in root.rglob("status.json"):
            status = json.loads(path.read_text())
            if status.get("study_id") == study["study_id"]:
                attempts.update(status.get("conditions", {}))
    paths = sorted({p.resolve() for root in roots for p in root.rglob("result.json")})
    with phase(logger, "verifying transcript results") as started:
        for completed, path in enumerate(paths, 1):
            candidate = json.loads(path.read_text())
            if candidate.get("study_id") != study["study_id"] or candidate["condition"]["stage"] != stage:
                continue
            result = verify_result(context, path.parent)
            identity = result["condition"]["condition_id"]
            if identity in results:
                raise ValueError("Duplicate successful transcript conditions exist across result roots")
            environment = json.loads((path.parent / "environment.json").read_text())
            versions = environment.get("dependency_versions")
            if not versions or (dependencies is not None and dependencies != versions):
                raise ValueError("Transcript dependency versions are missing or differ across systems")
            dependencies = versions
            if stage == "final":
                table = pd.read_csv(path.parent / "predictions_test.csv", dtype={"participant_id": str})
                task = result["condition"]["task"]
                target = "y_true" if task == "classification" else "phq8_true"
                if table[target].notna().any():
                    targets = dict(zip(table["participant_id"], table[target], strict=True))
                    if task in test_targets and test_targets[task] != targets:
                        raise ValueError("Transcript test targets differ across results")
                    test_targets[task] = targets
            results[identity] = {"result": result, "path": path.parent, "environment": environment, "digest": file_hash(path)}
            log_progress(logger, "verifying transcript results", completed, len(paths), started)
    return results, attempts


def _stats_context(context: tuple) -> tuple:
    study, participants, _, folds = context
    stats = deepcopy(study)
    stats["configuration"]["tabular_study"] = {
        "folds": len(folds["classification"]["folds"]),
        **stats["configuration"]["transcript_study"],
    }
    return stats, pd.DataFrame(participants), None, folds


def _runtime(rows: list[dict], results: dict, output: Path) -> None:
    for row in rows:
        environment = results[row["condition_id"]]["environment"]
        for key in ("cpu_threads", "container_ram_limit_mb"):
            row[key] = environment.get(key)
    table = pd.DataFrame(rows)
    table.to_csv(output / "condition_metrics.csv", index=False)
    if table.empty:
        table.to_csv(output / "runtime_summary.csv", index=False)
        return
    keys = [*GROUP_KEYS, "gpu_name", "gpu_vram_bytes", "cpu_threads", "container_ram_limit_mb"]
    metrics = [column for column in table if column.startswith("runtime/")]
    table.groupby(keys, dropna=False)[metrics].agg(["count", "mean", "std"]).to_csv(output / "runtime_summary.csv")


def _compare_tabular(context: tuple, oof: dict, report: Path, destination: Path, logger) -> None:
    study, participants, _, folds = context
    coverage = json.loads((report / "coverage.json").read_text())
    if coverage.get("study_id") != study["parent_study_id"] or coverage.get("stage") != "cv" or coverage.get("status") != "completed":
        raise ValueError("Tabular comparison requires the completed parent study CV report")
    summary = pd.read_csv(report / "cv_summary.csv")
    keys = list(GROUP_KEYS)
    settings = study["configuration"]["transcript_study"]
    expected_ids = {row["participant_id"] for row in participants if row["split"] == "train"}
    rows = []
    for identity, group in summary.groupby(keys, sort=True):
        family, modality, task, mode, seed = identity
        key = ("qwen35_2b", "transcript", task, mode, seed)
        if key not in oof:
            continue
        names = group["predictions_file"].unique()
        if len(names) != 1:
            raise ValueError("Tabular report has ambiguous pooled predictions")
        path = (report / names[0]).resolve()
        if not path.is_relative_to(report.resolve()):
            raise ValueError("Tabular report prediction path escapes its directory")
        left = pd.read_csv(path, dtype={"participant_id": str})
        right = oof[key]
        left, right = _aligned(left, right, task, folds=True)
        assignments = {p: fold["fold"] for fold in folds[task]["folds"] for p in fold["evaluation"]}
        if set(left["participant_id"]) != expected_ids or any(assignments[p] != fold for p, fold in zip(left["participant_id"], left["fold"], strict=True)):
            raise ValueError("Tabular report predictions do not match imported folds")
        name = f"comparing transcript with {family} {modality} {task} {mode}"
        with phase(logger, name) as started:
            differences = paired_bootstrap(left, right, task, iterations=settings["bootstrap_iterations"], seed=settings["bootstrap_seed"],
                                           progress=lambda n, total, name=name: log_progress(logger, name, n, total, started))
        for difference in differences:
            difference["tabular_value"] = difference.pop("frozen_icl")
            difference["transcript_value"] = difference.pop("fine_tuning")
            difference["delta_transcript_minus_tabular"] = difference.pop("delta_ft_minus_icl")
            rows.append({"tabular_family": family, "tabular_modality": modality, "transcript_family": "qwen35_2b",
                         "task": task, "mode": mode, "model_seed": seed,
                         "interpretation": "model and input both differ", **difference})
    pd.DataFrame(rows).to_csv(destination / "cross_input_comparisons.csv", index=False)


def collect_study(bundle: str | Path, roots: list[str | Path], destination: str | Path,
                  *, stage: str = "cv", allow_partial: bool = False, tabular_report: str | Path | None = None) -> Path:
    context = read_bundle(bundle)
    require_audit(bundle, context[0])
    if stage not in context[0]["conditions"] or (stage != "cv" and tabular_report is not None):
        raise ValueError("Use cv or final; optional tabular report comparisons require cv")
    roots = [Path(root).resolve() for root in roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("Transcript result roots must be existing directories")
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    logger = configure_run_logger(output / "collect.log", "INFO")
    report = {"study_id": context[0]["study_id"], "parent_study_id": context[0]["parent_study_id"],
              "stage": stage, "status": "collecting", "expected": len(context[0]["conditions"][stage]), "completed": 0}
    _json(output / "coverage.json", report)
    try:
        results, attempts = _discover(context, roots, stage, logger)
        coverage = _coverage(context[0], stage, results, attempts, [])
        missing = [row for row in coverage if row["status"] != "completed"]
        report.update(conditions=coverage, completed=len(results), status="incomplete" if missing else "collecting")
        _json(output / "coverage.json", report)
        pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
        if missing and not allow_partial:
            raise ValueError(f"Transcript campaign is incomplete: {len(missing)} conditions remain")
        _json(output / "software_provenance.json", {"software_identity": context[0]["software_identity"],
                                                   "parent_study_id": context[0]["parent_study_id"],
                                                   "result_hashes": {k: v["digest"] for k, v in results.items()}})
        with phase(logger, "summarizing transcript results"):
            rows = _write_condition_metrics(results, output, stage)
            _runtime(rows, results, output)
            if stage == "cv":
                stats_context = _stats_context(context)
                oof = _write_cv_summaries(stats_context, results, output)
                write_cv_comparisons(stats_context[0], results, oof, output, logger)
                if tabular_report:
                    _compare_tabular(context, oof, Path(tabular_report).resolve(), output, logger)
            else:
                write_final_comparisons(results, output)
        report["status"] = "incomplete" if missing else "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        _json(output / "coverage.json", report)
    return output
