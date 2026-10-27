from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.tabular_study.bundle import _json, file_hash


def write_result(context: tuple, condition: dict, output: Path) -> dict:
    study, _, _, _ = context
    names = [
        "study_run.json",
        "fold_assignments.json",
        "feature_selection.json",
        "config_resolved.yaml",
        "adaptation_metadata.json",
        "environment.json",
        "model.json",
        "runtime.json",
        "wandb_run.json",
    ]
    splits = ("cv",) if condition["stage"] == "cv" else ("dev", "test")
    for split in splits:
        names.append(f"predictions_{split}.csv")
        if (output / f"metrics_{split}.json").exists():
            names.append(f"metrics_{split}.json")
    detail = json.loads((output / "adaptation_metadata.json").read_text())
    checkpoint = None
    if condition["mode"] == "fine_tuning":
        path = Path(detail["checkpoint_path"]).resolve()
        root = (output / "checkpoints").resolve()
        if not path.is_relative_to(root) or file_hash(path) != detail["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint does not match its metadata")
        checkpoint = {"path": str(path.relative_to(output.resolve())), "sha256": file_hash(path)}
    result = {
        "version": 1,
        "study_id": study["study_id"],
        "condition": condition,
        "software_identity": study["software_identity"],
        "files": {name: file_hash(output / name) for name in names},
        "checkpoint": checkpoint,
    }
    _json(output / "result.json", result)
    try:
        verify_result(context, output, require_checkpoint=True)
    except BaseException:
        (output / "result.json").unlink(missing_ok=True)
        raise
    return result


def verify_result(context: tuple, output: str | Path, *, require_checkpoint: bool = False) -> dict:
    study, table, _, folds = context
    output = Path(output).resolve()
    result = json.loads((output / "result.json").read_text())
    if (
        result.get("version") != 1
        or result.get("study_id") != study["study_id"]
        or result.get("software_identity") != study["software_identity"]
    ):
        raise ValueError("Result study or software identity mismatch")
    condition = result["condition"]
    if condition not in study["conditions"][condition["stage"]]:
        raise ValueError("Result condition is outside the prepared schedule")
    mandatory = {
        "study_run.json",
        "fold_assignments.json",
        "feature_selection.json",
        "config_resolved.yaml",
        "adaptation_metadata.json",
        "environment.json",
        "model.json",
        "runtime.json",
        "wandb_run.json",
    }
    splits = ("cv",) if condition["stage"] == "cv" else ("dev", "test")
    mandatory.update(f"predictions_{split}.csv" for split in splits)
    mandatory.add("metrics_cv.json" if condition["stage"] == "cv" else "metrics_dev.json")
    if not mandatory <= set(result["files"]):
        raise ValueError("Result is missing required artifacts")
    for name, digest in result["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Result artifact is missing or changed")
    checkpoint = result.get("checkpoint")
    if condition["mode"] == "fine_tuning":
        if not checkpoint:
            raise ValueError("Result is missing checkpoint provenance")
        path = (output / checkpoint["path"]).resolve()
        if (
            not path.is_relative_to(output / "checkpoints")
            or (require_checkpoint and not path.is_file())
            or (path.exists() and file_hash(path) != checkpoint["sha256"])
        ):
            raise ValueError("Result checkpoint is missing or changed")
    elif checkpoint is not None:
        raise ValueError("Frozen ICL cannot contain a fine-tuned checkpoint")
    record = json.loads((output / "study_run.json").read_text())
    if (
        record.get("status") != "completed"
        or record.get("condition_id") != condition["condition_id"]
        or record.get("stage") != condition["stage"]
        or record.get("software_identity") != study["software_identity"]
        or record.get("dataset_id") != study["configuration"]["data"]["dataset_id"]
        or record.get("task") != condition["task"]
        or record.get("mode") != condition["mode"]
        or record.get("fold") != condition["fold"]
        or record.get("model_seed") != condition["model_seed"]
        or record.get("selection_labels_used") != (condition["mode"] == "fine_tuning")
    ):
        raise ValueError("Result lifecycle provenance mismatch")
    saved_config = yaml.safe_load((output / "config_resolved.yaml").read_text())
    expected_model = study["configuration"]["profiles"][
        f"{condition['family']}:{condition['task']}:{condition['mode']}"
    ]
    expected_model = {
        **expected_model,
        "parameters": {**expected_model["parameters"], "random_state": condition["model_seed"]},
    }
    if saved_config["model"] != expected_model:
        raise ValueError("Result model settings differ from the prepared condition")
    assignments = json.loads((output / "fold_assignments.json").read_text())
    partition = (
        folds[condition["task"]]["final"]
        if condition["stage"] == "final"
        else next(x for x in folds[condition["task"]]["folds"] if x["fold"] == condition["fold"])
    )
    expected_roles = {key: partition[key] for key in ("fit", "selection")}
    if condition["stage"] == "cv":
        expected_roles["evaluation"] = partition["evaluation"]
    else:
        expected_roles.update(
            {
                split: sorted(
                    table.loc[table["split"] == split, "participant_id"].astype(str).tolist()
                )
                for split in splits
            }
        )
    if (
        assignments["roles"] != expected_roles
        or assignments["manifest_id"] != fingerprint(partition)
        or record.get("manifest_id") != fingerprint(partition)
    ):
        raise ValueError("Result participant assignments differ from the prepared folds")
    indexed = table.assign(participant_id=table["participant_id"].astype(str)).set_index(
        "participant_id"
    )
    for split in splits:
        predictions = pd.read_csv(
            output / f"predictions_{split}.csv", dtype={"participant_id": str}
        )
        ids = expected_roles["evaluation" if split == "cv" else split]
        if (
            predictions["participant_id"].duplicated().any()
            or set(predictions["participant_id"]) != set(ids)
            or set(predictions["split"]) != {split}
        ):
            raise ValueError("Result predictions do not match evaluation participants")
        columns = (
            ["y_pred", "prob_depressed"] if condition["task"] == "classification" else ["phq8_pred"]
        )
        if not np.isfinite(predictions[columns].to_numpy(dtype=float)).all():
            raise ValueError("Result predictions must be finite")
        if (
            condition["task"] == "classification"
            and not predictions["prob_depressed"].between(0, 1).all()
        ):
            raise ValueError("Result probabilities must be within [0,1]")
        if split != "test":
            target = "y_true" if condition["task"] == "classification" else "phq8_true"
            source = "target_binary" if condition["task"] == "classification" else "target_phq8"
            actual = predictions[target].to_numpy(dtype=float)
            expected = indexed.loc[predictions["participant_id"], source].to_numpy(dtype=float)
            if not np.array_equal(actual, expected):
                raise ValueError("Result targets differ from the prepared dataset")
    return result
