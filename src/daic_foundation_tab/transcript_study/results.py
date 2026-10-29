from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.tabular_study.bundle import _json, file_hash
from daic_foundation_tab.tabular_study.collection import _metrics, _scalars

from .bundle import condition_config, roles_for

REQUIRED = {"study_run.json", "fold_assignments.json", "prompt_audit.json", "config_resolved.yaml",
            "environment.json", "model_metadata.json", "runtime.json", "wandb_run.json", "run_identity.json",
            "participant_manifest.json", "demonstrations.csv"}


def write_result(context: tuple, condition: dict, output: Path) -> dict:
    study = context[0]
    files = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.suffix in {".json", ".yaml", ".csv", ".jsonl"}
             and p.name != "result.json"}
    adapter = output / "checkpoints" / "best_adapter"
    checkpoints = {str(p.relative_to(output)): file_hash(p) for p in sorted(adapter.rglob("*")) if p.is_file()}
    result = {"version": 1, "kind": "transcript", "study_id": study["study_id"],
              "parent_study_id": study["parent_study_id"], "software_identity": study["software_identity"],
              "condition": condition, "files": files, "checkpoints": checkpoints}
    _json(output / "result.json", result)
    try:
        verify_result(context, output, require_checkpoint=True)
    except BaseException:
        (output / "result.json").unlink(missing_ok=True)
        raise
    return result


def verify_result(context: tuple, output: str | Path, *, require_checkpoint: bool = False) -> dict:
    study, rows, _, folds = context
    output = Path(output).resolve()
    result = json.loads((output / "result.json").read_text())
    condition = result["condition"]
    if (result.get("kind") != "transcript" or result.get("version") != 1
            or result.get("study_id") != study["study_id"] or result.get("parent_study_id") != study["parent_study_id"]
            or result.get("software_identity") != study["software_identity"]
            or condition not in study["conditions"].get(condition["stage"], [])):
        raise ValueError("Transcript result identity mismatch")
    splits = ("cv",) if condition["stage"] == "cv" else ("dev", "test")
    required = REQUIRED | {f"predictions_{split}.csv" for split in splits} | {f"metrics_{splits[0]}.json"}
    if not required <= set(result["files"]):
        raise ValueError("Transcript result is missing required artifacts")
    for name, digest in result["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Transcript result artifact is missing or changed")
    checkpoints = result["checkpoints"]
    if (condition["mode"] == "fine_tuning") != bool(checkpoints):
        raise ValueError("Transcript adapter checkpoint provenance is invalid")
    if checkpoints and not any(name.endswith("adapter_config.json") for name in checkpoints):
        raise ValueError("Selected adapter configuration is missing")
    for name, digest in checkpoints.items():
        path = (output / name).resolve()
        if (not path.is_relative_to(output / "checkpoints/best_adapter")
                or (require_checkpoint and not path.is_file()) or (path.exists() and file_hash(path) != digest)):
            raise ValueError("Transcript adapter checkpoint is missing or changed")
    config = yaml.safe_load((output / "config_resolved.yaml").read_text())
    expected = condition_config(study["configuration"], condition["task"], condition["mode"], condition["model_seed"])
    if (config["model"] != expected["model"] or config["project"]["seed"] != condition["model_seed"]
            or config["evaluation"] != expected["evaluation"]
            or any(config["experiment"].get(k) != condition[k] for k in ("task", "mode"))):
        raise ValueError("Executed transcript settings differ from the prepared condition")
    record = json.loads((output / "study_run.json").read_text())
    for key, expected_value in {"status": "completed", "condition_id": condition["condition_id"],
                                "transcript_study_id": study["study_id"], "parent_study_id": study["parent_study_id"],
                                "software_identity": study["software_identity"], "task": condition["task"],
                                "mode": condition["mode"], "stage": condition["stage"], "fold": condition["fold"],
                                "model_seed": condition["model_seed"], "dataset_id": study["configuration"]["data"]["dataset_id"],
                                "selection_labels_used": condition["mode"] == "fine_tuning"}.items():
        if record.get(key) != expected_value:
            raise ValueError("Transcript lifecycle provenance mismatch")
    metadata = json.loads((output / "model_metadata.json").read_text())
    if metadata["checkpoint_revision"] != expected["model"]["revision"]:
        raise ValueError("Transcript model checkpoint revision differs")
    tracking = json.loads((output / "wandb_run.json").read_text())
    if config["tracking"]["wandb"]["enabled"] and (
        tracking.get("status") != "completed" or not tracking.get("run_id")
    ):
        raise ValueError("Transcript W&B tracking did not finish")
    roles = roles_for(rows, folds, condition)
    assignments = json.loads((output / "fold_assignments.json").read_text())
    if assignments != {"roles": roles, "manifest_id": fingerprint(roles)}:
        raise ValueError("Transcript assignments differ from imported folds")
    if record.get("manifest_id") != fingerprint(roles):
        raise ValueError("Transcript manifest identity differs")
    manifest = json.loads((output / "participant_manifest.json").read_text())
    if manifest["manifest_id"] != fingerprint(roles) or {
        role: sorted(row["participant_id"] for row in manifest["participants"] if row["role"] == role) for role in roles
    } != roles:
        raise ValueError("Transcript participant manifest differs")
    demos = pd.read_csv(output / "demonstrations.csv", dtype=str)["participant_id"].tolist()
    audit = json.loads((output / "prompt_audit.json").read_text())
    if demos != audit["demonstrations"] or not set(demos) <= set(roles["fit"]) or len(demos) != (2 if condition["mode"] == "frozen_icl" else 0):
        raise ValueError("Transcript demonstrations differ from the audited fit pool")
    indexed = {row["participant_id"]: row for row in rows}
    target, source, columns = (("y_true", "target_binary", ["y_pred", "prob_depressed"])
                                if condition["task"] == "classification" else ("phq8_true", "target_phq8", ["phq8_pred"]))
    for split in splits:
        table = pd.read_csv(output / f"predictions_{split}.csv", dtype={"participant_id": str})
        ids = roles["evaluation" if split == "cv" else split]
        if table["participant_id"].duplicated().any() or set(table["participant_id"]) != set(ids) or set(table["split"]) != {split}:
            raise ValueError("Transcript predictions do not match the evaluation participants")
        if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
            raise ValueError("Transcript predictions must be finite")
        if condition["task"] == "classification":
            if not table["prob_depressed"].between(0, 1).all() or not np.array_equal(table["y_pred"], (table["prob_depressed"] > .5).astype(int)):
                raise ValueError("Invalid transcript classification probabilities or decisions")
        elif not table["phq8_pred"].between(0, 24).all():
            raise ValueError("Transcript score is outside [0,24]")
        if split != "test" and not np.array_equal(table[target], [indexed[p][source] for p in table["participant_id"]]):
            raise ValueError("Transcript evaluation targets differ")
        metric_path = output / f"metrics_{split}.json"
        if metric_path.is_file():
            if table[target].isna().any():
                raise ValueError("Scored transcript predictions require complete targets")
            actual, expected_metrics = _scalars(json.loads(metric_path.read_text())), _scalars(_metrics(condition["task"], table))
            if actual.keys() != expected_metrics.keys() or any(not np.isclose(actual[k], v) for k, v in expected_metrics.items()):
                raise ValueError("Transcript metrics differ from saved predictions")
    return result
