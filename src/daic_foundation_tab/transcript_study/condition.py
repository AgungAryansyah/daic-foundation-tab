from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.tabular_study.bundle import _json
from daic_foundation_tab.text_runner import run_text_experiment

from .audit import require_audit
from .bundle import condition_config, dataset_for, read_bundle, roles_for


def execution_config(study: dict, condition: dict, host: dict) -> dict:
    config = condition_config(study["configuration"], condition["task"], condition["mode"], condition["model_seed"])
    config["project"].update(output_root=str(Path(host["output_root"]) / condition["condition_id"]),
                             cache_root=host.get("cache_root", str(Path(host["output_root"]) / "cache")))
    config["data"]["root"] = "."
    config["data"]["test_ground_truth"] = host.get("test_ground_truth")
    config["experiment"]["name"] = condition["condition_id"][:16]
    config["tracking"]["wandb"].update(host.get("wandb", {}))
    tracking = config["tracking"]["wandb"]
    tracking["enabled"] = tracking.get("mode", "online") != "disabled"
    tracking["group"] = study["study_id"]
    config["study"] = {
        "transcript_study_id": study["study_id"], "parent_study_id": study["parent_study_id"],
        "software_identity": study["software_identity"], "condition_id": condition["condition_id"],
        "stage": condition["stage"], "fold": condition["fold"], "model_seed": condition["model_seed"],
        "host_label": host["host_label"], "execution_group": f"{condition['family']}-{condition['task']}",
        "cpu_threads": host.get("cpu_threads", 1),
    }
    return config


def run_condition(bundle: str | Path, condition: dict, host: dict, *, resume: Path | None = None,
                  extra_environment: dict | None = None, runtime_metrics=None) -> Path:
    context = read_bundle(bundle)
    study, rows, _, folds = context
    if condition not in study["conditions"].get(condition["stage"], []):
        raise ValueError("Transcript condition is outside the prepared schedule")
    audit = require_audit(bundle, study)["conditions"][condition["condition_id"]]
    config = execution_config(study, condition, host)
    roles = roles_for(rows, folds, condition)
    config["study"].update(manifest_id=fingerprint(roles),
                           cohort_id=fingerprint(sorted(row["participant_id"] for row in rows)))
    if resume is not None:
        config["_resume_directory"] = str(resume)
    dataset = dataset_for(context, condition)
    evaluation_splits = {"cv": "evaluation"} if condition["stage"] == "cv" else {"dev": "dev", "test": "test"}
    completed = resume is not None and (resume / "study_run.json").is_file() and json.loads(
        (resume / "study_run.json").read_text()
    ).get("status") == "completed"
    if completed:
        record = json.loads((resume / "study_run.json").read_text())
        if any(record.get(key) != config["study"][key] for key in ("condition_id", "software_identity", "transcript_study_id")):
            raise ValueError("Incomplete result belongs to a different transcript condition")
        output = resume
    else:
        output = run_text_experiment(config, dataset=dataset, evaluation_splits=evaluation_splits,
                                     extra_environment=extra_environment, runtime_metrics=runtime_metrics)
    actual = pd.read_csv(output / "demonstrations.csv", dtype=str)["participant_id"].tolist()
    if actual != audit["demonstrations"]:
        raise ValueError("Executed demonstrations differ from the audited condition")
    _json(output / "fold_assignments.json", {"roles": roles, "manifest_id": fingerprint(roles)})
    _json(output / "prompt_audit.json", deepcopy(audit))
    return output
