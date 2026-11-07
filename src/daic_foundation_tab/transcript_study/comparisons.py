from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daic_foundation_tab.tabular_study.comparisons import paired_bootstrap
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

from .bundle import MODES, TASKS, read_bundle


def _read_report(study: dict, directory: Path, mode: str) -> pd.DataFrame:
    conditions = study["conditions"]["cv"]
    expected = {"study_id": study["study_id"], "parent_study_id": study["parent_study_id"],
                "stage": "cv", "status": "completed", "expected": len(conditions),
                "completed": len(conditions),
                "conditions": [{**row, "status": "completed"} for row in conditions]}
    coverage = json.loads((directory / "coverage.json").read_text())
    if any(coverage.get(key) != value for key, value in expected.items()):
        raise ValueError("Transcript comparison requires a completed report for its bundle")
    if mode not in MODES or mode not in {row["mode"] for row in conditions}:
        raise ValueError("Transcript comparison method is absent from the bundle")
    summary = pd.read_csv(directory / "cv_summary.csv")
    return summary.loc[(summary["family"] == study["configuration"]["model"]["name"])
                       & (summary["modality"] == "transcript") & (summary["mode"] == mode)]


def _predictions(directory: Path, summary: pd.DataFrame, task: str, seed: int,
                 participants: list, folds: dict, metric: str) -> pd.DataFrame:
    rows = summary.loc[(summary["task"] == task) & (summary["model_seed"] == seed)]
    names = rows["predictions_file"].unique()
    if len(names) != 1 or sum(rows["metric"] == metric) != 1:
        raise ValueError("Transcript report lacks unique pooled predictions")
    path = (directory / names[0]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Transcript report prediction path escapes its directory")
    table = pd.read_csv(path, dtype={"participant_id": str}).set_index("participant_id")
    target = "y_true" if task == "classification" else "phq8_true"
    field = "target_binary" if task == "classification" else "target_phq8"
    targets = {row["participant_id"]: row[field] for row in participants if row["split"] == "train"}
    assignments = {p: fold["fold"] for fold in folds[task]["folds"] for p in fold["evaluation"]}
    if (not table.index.is_unique or set(table.index) != set(targets)
            or table["fold"].to_dict() != assignments or table[target].to_dict() != targets):
        raise ValueError("Transcript predictions do not match imported participants, targets, and folds")
    return table.reset_index()


def compare_cv(reference: tuple, candidate: tuple, destination: str | Path) -> Path:
    specs = [(Path(bundle), Path(report).resolve(), mode) for bundle, report, mode in (reference, candidate)]
    contexts = [read_bundle(bundle, verify_software=False) for bundle, _, _ in specs]
    studies = [context[0] for context in contexts]
    left, right = studies
    if any(left[key] != right[key] for key in ("parent_study_id", "files")):
        raise ValueError("Transcript comparison requires matching parent and transcript inputs")
    models = [study["configuration"]["model"] for study in studies]
    if models[0]["name"] == models[1]["name"] and models[0]["revision"] != models[1]["revision"]:
        raise ValueError("Transcript checkpoint revisions differ for the same model")
    settings = [study["configuration"]["transcript_study"] for study in studies]
    if any(settings[0][key] != settings[1][key] for key in ("model_seeds", "bootstrap_iterations", "bootstrap_seed")):
        raise ValueError("Transcript model seeds or bootstrap settings differ")
    summaries = [_read_report(study, directory, mode)
                 for study, (_, directory, mode) in zip(studies, specs, strict=True)]
    metadata = {}
    for name, model, study, (_, _, mode) in zip(("reference", "candidate"), models, studies, specs, strict=True):
        metadata.update({f"{name}_family": model["name"], f"{name}_revision": model["revision"],
                         f"{name}_study_id": study["study_id"], f"{name}_adaptation":
                         model["parameters"].get("adapter_method", "lora") if mode == "fine_tuning" else mode})
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    logger = configure_run_logger(output / "compare.log", "INFO")
    _, participants, _, folds = contexts[0]
    rows = []
    for task, metric in zip(TASKS, ("roc_auc", "mae"), strict=True):
        for seed in settings[0]["model_seeds"]:
            tables = [_predictions(directory, summary, task, seed, participants, folds, metric)
                      for (_, directory, _), summary in zip(specs, summaries, strict=True)]
            name = f"bootstrapping {task} seed {seed}"
            with phase(logger, name) as started:
                differences = paired_bootstrap(*tables, task, iterations=settings[0]["bootstrap_iterations"],
                                               seed=settings[0]["bootstrap_seed"],
                                               progress=lambda n, total: log_progress(logger, name, n, total, started))
            primary = next(row for row in differences if row["metric"] == metric)
            primary["reference"] = primary.pop("frozen_icl")
            primary["candidate"] = primary.pop("fine_tuning")
            primary["delta_candidate_minus_reference"] = primary.pop("delta_ft_minus_icl")
            rows.append({**metadata, "task": task, "model_seed": seed, **primary})
    path = output / "paired_transcript_cv.csv"
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)
    return path
