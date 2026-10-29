from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daic_foundation_tab.data.participants import fingerprint
from daic_foundation_tab.models.qwen import load_text_tokenizer
from daic_foundation_tab.models.text_prompt import PromptEncoder, select_demonstrations
from daic_foundation_tab.tabular_study.bundle import _json
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

from .bundle import condition_config, dataset_for, read_bundle


def audit_bundle(directory: str | Path, *, tokenizer=None) -> dict:
    directory = Path(directory).resolve()
    context = read_bundle(directory)
    study = context[0]
    config = study["configuration"]
    logger = configure_run_logger(directory / "audit.log", "INFO")
    with phase(logger, "loading transcript tokenizer"):
        tokenizer = tokenizer if tokenizer is not None else load_text_tokenizer(config["model"])
    report = {"version": 1, "study_id": study["study_id"], "conditions": {}}
    conditions = [x for stage in study["conditions"].values() for x in stage]
    with phase(logger, "auditing complete prompts") as started:
        log_progress(logger, "auditing complete prompts", 0, len(conditions), started)
        for completed, condition in enumerate(conditions, 1):
            dataset = dataset_for(context, condition)
            settings = condition_config(config, condition["task"], condition["mode"], condition["model_seed"])
            encoder = PromptEncoder(tokenizer, condition["task"], settings["model"]["parameters"],
                                    {"model_id": config["model"]["model_id"], "revision": config["model"]["revision"]})
            demonstrations = select_demonstrations(
                dataset.examples("fit", labeled=True), settings["model"]["parameters"]["demonstrations"],
                condition["model_seed"],
            )
            encoder.demonstrations = demonstrations
            roles = ["evaluation"] if condition["stage"] == "cv" else ["dev", "test"]
            if condition["mode"] == "fine_tuning":
                roles = ["fit", "selection", *roles]
            records, errors = [], []
            for role in roles:
                labeled = role in {"fit", "selection"}
                for example in dataset.examples(role, labeled=labeled):
                    try:
                        tokens = encoder.prompt(example)
                        total = len(tokens) + encoder.max_new_tokens
                        if labeled:
                            total = max(total, len(encoder.training_record(example)["input_ids"]))
                        if condition["task"] == "classification":
                            for answer in ("0", "1"):
                                encoder.check_length(len(tokens), len(encoder.completion(answer)))
                        records.append({"participant_id": example.participant_id, "role": role,
                                        "prompt_tokens": len(tokens), "total_tokens": total})
                    except ValueError as error:
                        errors.append({"participant_id": example.participant_id, "role": role, "error": str(error)})
            lengths = [row["prompt_tokens"] for row in records]
            report["conditions"][condition["condition_id"]] = {
                "status": "failed" if errors else "passed", "demonstrations": [x.participant_id for x in demonstrations],
                "records": records, "errors": errors,
                "statistics": {"count": len(lengths), "min": min(lengths) if lengths else None,
                               "max": max(lengths) if lengths else None,
                               "p50": float(np.median(lengths)) if lengths else None,
                               "p95": float(np.quantile(lengths, .95)) if lengths else None},
            }
            log_progress(logger, "auditing complete prompts", completed, len(conditions), started)
    report["status"] = "passed" if all(x["status"] == "passed" for x in report["conditions"].values()) else "failed"
    report["audit_id"] = fingerprint(report)
    _json(directory / "audit.json", report)
    if report["status"] != "passed":
        raise ValueError("Transcript prompt audit failed; inspect private audit.json. No truncation was applied")
    return report


def require_audit(directory: str | Path, study: dict) -> dict:
    path = Path(directory) / "audit.json"
    if not path.is_file():
        raise ValueError("Transcript campaign requires a complete prompt audit")
    report = json.loads(path.read_text())
    payload = {key: value for key, value in report.items() if key != "audit_id"}
    expected = {x["condition_id"] for stage in study["conditions"].values() for x in stage}
    if (report.get("version") != 1 or report.get("study_id") != study["study_id"]
            or report.get("status") != "passed" or set(report["conditions"]) != expected
            or any(x.get("status") != "passed" for x in report["conditions"].values())
            or report.get("audit_id") != fingerprint(payload)):
        raise ValueError("Transcript prompt audit is failed, incomplete, or changed")
    return report
