from __future__ import annotations

import json
import shutil
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd

from daic_foundation_tab.config import load_config, validate_config
from daic_foundation_tab.data.participants import fingerprint, source_matches
from daic_foundation_tab.data.transcripts import TranscriptDataset, read_transcript
from daic_foundation_tab.tabular_study.bundle import (
    BUNDLE_FILES,
    _json,
    file_hash,
    software_identity,
)
from daic_foundation_tab.tabular_study.bundle import (
    verify_bundle as verify_tabular,
)
from daic_foundation_tab.tracking.logging import configure_run_logger, log_progress, phase

FILES = ("participants.json", "folds.json", "transcripts.json", "sources.json")
TASKS = ("classification", "regression")
MODES = ("frozen_icl", "fine_tuning")


def condition_config(config: dict, task: str, mode: str, seed: int) -> dict:
    config = deepcopy(config)
    config["project"]["seed"] = seed
    config["experiment"].update(task=task, mode=mode)
    config["evaluation"]["threshold"] = 0.5 if task == "classification" else None
    parameters = config["model"]["parameters"]
    parameters["demonstrations"] = config["transcript_study"]["demonstrations"] if mode == "frozen_icl" else 0
    if mode == "frozen_icl":
        parameters.pop("adapter_method", None)
    return config


def load_settings(path: str | Path) -> dict:
    config = load_config(path, _validate=False)
    config.pop("_config_path", None)
    settings = config.get("transcript_study", {})
    if not isinstance(settings.get("name"), str) or not settings["name"].strip():
        raise ValueError("transcript_study.name is required")
    seeds = settings.get("model_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("Transcript model_seeds must contain unique nonnegative integers")
    if settings.get("demonstrations") != 2:
        raise ValueError("The initial transcript study requires two demonstrations")
    methods = settings.get("methods", list(MODES))
    if (not isinstance(methods, list) or not methods or any(mode not in MODES for mode in methods)
            or len(methods) != len(set(methods))):
        raise ValueError("Transcript methods must contain unique values from frozen_icl and fine_tuning")
    if config["model"]["parameters"].get("adapter_method") == "dora" and "fine_tuning" not in methods:
        raise ValueError("DoRA requires a fine_tuning condition")
    for key in ("bootstrap_iterations", "bootstrap_seed"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key.endswith("iterations") else 0):
            raise ValueError(f"Invalid transcript {key}")
    if config["data"]["transcripts"].get("speaker_policy") != "participant_only":
        raise ValueError("Transcript campaigns require participant_only speech")
    revision = config["model"].get("revision", "")
    if not isinstance(revision, str) or len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
        raise ValueError("Transcript campaigns require an immutable checkpoint revision")
    for task, mode in product(TASKS, methods):
        validate_config(condition_config(config, task, mode, seeds[0]))
    return config


def _logical(config: dict) -> dict:
    config = deepcopy(config)
    for key in ("root", "test_ground_truth"):
        config["data"].pop(key, None)
    for key in ("output_root", "cache_root"):
        config["project"].pop(key, None)
    return config


def _schedule(study_id: str, config: dict, folds: dict) -> dict:
    result = {}
    settings = config["transcript_study"]
    for stage in ("cv", "final"):
        conditions = []
        for task, mode, seed in product(TASKS, settings.get("methods", MODES), settings["model_seeds"]):
            for fold in folds[task]["folds"] if stage == "cv" else [None]:
                condition = {"stage": stage, "family": config["model"]["name"], "modality": "transcript", "task": task,
                                 "mode": mode, "model_seed": seed, "fold": fold["fold"] if fold else None}
                conditions.append({**condition, "condition_id": fingerprint({"study_id": study_id, **condition})})
        result[stage] = conditions
    return result


def _identity(config: dict, parent: dict, files: dict, software: str) -> str:
    return fingerprint({"configuration": config, "parent_study_id": parent["study_id"],
                            "files": files, "software_identity": software})


def prepare_bundle(config_path: str | Path, tabular: str | Path, destination: str | Path,
                   *, tokenizer=None) -> Path:
    config = load_settings(config_path)
    tabular, destination = Path(tabular).resolve(), Path(destination).resolve()
    parent = verify_tabular(tabular, verify_software=False)
    if parent["configuration"]["data"]["dataset_id"] != config["data"]["dataset_id"]:
        raise ValueError("Transcript and tabular dataset identities differ")
    if set(parent["conditions"]) != {"cv", "final"}:
        raise ValueError("Parent study requires CV and final assignments")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(destination.iterdir()):
        raise ValueError("Use an empty transcript study destination")
    destination.chmod(0o700)
    logger = configure_run_logger(destination / "prepare.log", "INFO")
    participants = pd.read_parquet(tabular / "features.parquet")
    participants = participants[["participant_id", "split", "target_binary", "target_phq8"]]
    participants["participant_id"] = participants["participant_id"].astype(str)
    participants = participants.sort_values("participant_id")
    if participants["participant_id"].duplicated().any():
        raise ValueError("Duplicate parent participants")
    participants.loc[participants["split"] == "test", ["target_binary", "target_phq8"]] = float("nan")
    rows = json.loads(participants.to_json(orient="records"))
    root = Path(config["data"]["root"]).expanduser().resolve()
    settings = config["data"]["transcripts"]
    texts, sources = {}, {}
    with phase(logger, "preparing transcripts") as started:
        log_progress(logger, "preparing transcripts", 0, len(rows), started)
        for completed, row in enumerate(rows, 1):
            participant = row["participant_id"]
            matches = source_matches(root, settings["pattern"], participant)
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one transcript for participant {participant}")
            before = file_hash(matches[0])
            texts[participant] = read_transcript(matches[0], settings)
            if before != file_hash(matches[0]):
                raise ValueError("Transcript source changed during preparation")
            sources[participant] = {"source_sha256": before, "text_identity": fingerprint(texts[participant])}
            log_progress(logger, "preparing transcripts", completed, len(rows), started)
    parent_dir = destination / "parent"
    parent_dir.mkdir(mode=0o700)
    for name in (*BUNDLE_FILES, "study.json", "execution_revisions.json"):
        if (tabular / name).is_file():
            shutil.copyfile(tabular / name, parent_dir / name)
    copied = verify_tabular(parent_dir, verify_software=False)
    if copied != parent:
        raise ValueError("Parent study changed during preparation")
    _json(destination / "participants.json", rows)
    shutil.copyfile(tabular / "folds.json", destination / "folds.json")
    _json(destination / "transcripts.json", texts)
    _json(destination / "sources.json", sources)
    logical = _logical(config)
    files = {name: file_hash(destination / name) for name in FILES}
    software = software_identity()
    study_id = _identity(logical, parent, files, software)
    study = {"version": 1, "kind": "transcript", "study_id": study_id, "parent_study_id": parent["study_id"],
                 "software_identity": software, "configuration": logical, "files": files,
                 "conditions": _schedule(study_id, logical, json.loads((destination / "folds.json").read_text()))}
    _json(destination / "study.json", study)
    from .audit import audit_bundle

    audit_bundle(destination, tokenizer=tokenizer)
    return destination


def read_bundle(directory: str | Path, *, verify_software: bool = True) -> tuple:
    directory = Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    if study.get("version") != 1 or study.get("kind") != "transcript" or set(study["files"]) != set(FILES):
        raise ValueError("Unsupported transcript bundle")
    if any(file_hash(directory / name) != digest for name, digest in study["files"].items()):
        raise ValueError("Transcript bundle content changed")
    parent = verify_tabular(directory / "parent", verify_software=False)
    config = study["configuration"]
    folds = json.loads((directory / "folds.json").read_text())
    if (study["parent_study_id"] != parent["study_id"]
            or file_hash(directory / "folds.json") != parent["files"]["folds.json"]
            or study["study_id"] != _identity(config, parent, study["files"], study["software_identity"])
            or study["conditions"] != _schedule(study["study_id"], config, folds)):
        raise ValueError("Transcript study identity or imported folds changed")
    if verify_software and study["software_identity"] != software_identity():
        raise ValueError("Transcript source or dependency identity changed")
    rows = json.loads((directory / "participants.json").read_text())
    texts = json.loads((directory / "transcripts.json").read_text())
    if {row["participant_id"] for row in rows} != set(texts):
        raise ValueError("Transcript cohort differs from imported participants")
    return study, rows, texts, folds


def roles_for(rows: list, folds: dict, condition: dict) -> dict:
    assignments = folds[condition["task"]]
    partition = assignments["final"] if condition["stage"] == "final" else next(
        x for x in assignments["folds"] if x["fold"] == condition["fold"]
    )
    roles = {key: partition[key] for key in ("fit", "selection")}
    if condition["stage"] == "cv":
        roles["evaluation"] = partition["evaluation"]
    else:
        roles.update({split: sorted(row["participant_id"] for row in rows if row["split"] == split)
                      for split in ("dev", "test")})
    ids = [participant for values in roles.values() for participant in values]
    indexed = {row["participant_id"]: row for row in rows}
    if len(set(ids)) != len(ids) or not all(roles.values()) or not set(ids) <= set(indexed):
        raise ValueError("Invalid or overlapping transcript fold roles")
    for role in ("fit", "selection", "evaluation"):
        if any(indexed[p]["split"] != "train" for p in roles.get(role, [])):
            raise ValueError("CV fit, selection, and evaluation require official training participants")
    return roles


def dataset_for(context: tuple, condition: dict) -> TranscriptDataset:
    study, rows, texts, folds = context
    roles = roles_for(rows, folds, condition)
    indexed = {row["participant_id"]: row for row in rows}
    participants = [{**indexed[p], "role": role} for role, ids in roles.items() for p in ids]
    manifest = {"task": condition["task"], "participants": participants, "manifest_id": fingerprint(roles)}
    return TranscriptDataset(texts, manifest, fingerprint({"study_id": study["study_id"], "manifest": manifest}))
