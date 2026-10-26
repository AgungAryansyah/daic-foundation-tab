from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .labels import load_official_labels


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_matches(root: Path, pattern: str, participant_id: str) -> list[Path]:
    return sorted(
        path
        for path in root.glob(pattern.format(participant_id=participant_id))
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def participant_index(config: dict, progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    data = config["data"]
    if not data.get("dataset_id"):
        raise ValueError("data.dataset_id is required for shared participant manifests")
    table = load_official_labels(data).table.copy()
    table.insert(0, "dataset_id", data["dataset_id"])
    root = Path(data["root"]).expanduser().resolve()
    requested = config.get("study", {}).get(
        "required_inputs", [config.get("input", {}).get("kind", "tabular")]
    )
    if not requested or not set(requested) <= {"tabular", "transcript"}:
        raise ValueError("study.required_inputs must select tabular and/or transcript")
    reasons = [[] for _ in range(len(table))]
    total = len(table) * len(requested)
    completed = 0
    if progress:
        progress(0, total)
    for kind in requested:
        if kind == "transcript":
            patterns = [data["transcripts"]["pattern"]]
        else:
            modalities = {
                "audio": {"audio"},
                "visual": {"visual"},
                "audio_visual": {"audio", "visual"},
            }[
                config.get("study", {}).get(
                    "feature_set", config["experiment"].get("feature_set", "audio_visual")
                )
            ]
            patterns = [
                item["pattern"]
                for item in data.get("source_groups", {}).values()
                if item["modality"] in modalities
            ]
            if not patterns:
                raise ValueError("Shared tabular cohort requires source_groups")
        available = []
        for index, participant_id in enumerate(table["participant_id"]):
            matches = [source_matches(root, pattern, participant_id) for pattern in patterns]
            if any(len(paths) > 1 for paths in matches):
                raise ValueError(f"Ambiguous {kind} source for participant {participant_id}")
            present = all(len(paths) == 1 and paths[0].stat().st_size > 0 for paths in matches)
            available.append(present)
            if not present:
                reasons[index].append(f"missing {kind}")
            completed += 1
            if progress:
                progress(completed, total)
        table[f"has_{kind}"] = available
    table["included"] = [not reason for reason in reasons]
    table["exclusion_reason"] = ["; ".join(reason) for reason in reasons]
    return table


def prepare_manifest(config: dict, progress: Callable[[int, int], None] | None = None) -> dict:
    index = participant_index(config, progress)
    table = index.loc[index["included"]].copy().reset_index(drop=True)
    if any(not (table["split"] == split).any() for split in ("train", "dev", "test")):
        raise ValueError("The selected cohort must contain train, dev, and test participants")
    task = config["experiment"]["task"]
    seed = int(
        config["evaluation"]["fine_tuning"].get("validation_seed", config["project"]["seed"])
    )
    train = table.loc[table["split"] == "train"]
    budget = config.get("study", {}).get("label_budget", len(train))
    if isinstance(budget, bool) or not isinstance(budget, int) or not 2 <= budget <= len(train):
        raise ValueError("label_budget must include fit and selection labels within training size")
    if budget < len(train):
        if task == "classification":
            selected, _ = next(
                StratifiedShuffleSplit(n_splits=1, train_size=budget, random_state=seed).split(
                    np.zeros(len(train)), train["target_binary"].astype(int)
                )
            )
        else:
            selected = np.random.default_rng(seed).permutation(len(train))[:budget]
        train = train.iloc[selected]
    from daic_foundation_tab.evaluation.fine_tuning import stratified_fine_tune_partition
    from daic_foundation_tab.evaluation.regression_splits import regression_fine_tune_partition

    partitioner = (
        stratified_fine_tune_partition
        if task == "classification"
        else regression_fine_tune_partition
    )
    column = "target_binary" if task == "classification" else "target_phq8"
    partition = partitioner(
        train[column].reset_index(drop=True),
        train["participant_id"].reset_index(drop=True),
        float(config["evaluation"]["fine_tuning"]["validation_fraction"]),
        seed,
        training_role="fit",
        validation_role="selection",
    )
    roles = partition.assignment.set_index("participant_id")["role"].to_dict()
    table["role"] = [
        roles.get(row.participant_id, row.split if row.split != "train" else "unused")
        for row in table.itertuples()
    ]
    records = json.loads(
        table[["participant_id", "split", "role", "target_binary", "target_phq8"]].to_json(
            orient="records"
        )
    )
    payload = {
        "version": 1,
        "dataset_id": config["data"]["dataset_id"],
        "task": task,
        "seed": seed,
        "label_budget": budget,
        "required_inputs": config.get("study", {}).get(
            "required_inputs", [config.get("input", {}).get("kind", "tabular")]
        ),
        "participants": records,
        "exclusions": json.loads(
            index.loc[~index["included"], ["participant_id", "exclusion_reason"]].to_json(
                orient="records"
            )
        ),
    }
    return {**payload, "manifest_id": fingerprint(payload)}


def load_manifest(path: str | Path, config: dict) -> dict:
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != fingerprint(payload):
        raise ValueError("Participant manifest fingerprint mismatch")
    if manifest["dataset_id"] != config["data"].get("dataset_id"):
        raise ValueError("Participant manifest dataset identity mismatch")
    if manifest["task"] != config["experiment"]["task"]:
        raise ValueError("Participant manifest task mismatch")
    table = pd.DataFrame(manifest["participants"])
    table["participant_id"] = table["participant_id"].astype(str)
    if table["participant_id"].duplicated().any():
        raise ValueError("Participant manifest contains duplicate or overlapping participants")
    permitted = {
        "fit": "train",
        "selection": "train",
        "unused": "train",
        "dev": "dev",
        "test": "test",
    }
    if any(permitted.get(row.role) != row.split for row in table.itertuples()):
        raise ValueError("Participant manifest roles leak across official splits")
    if int(table["role"].isin(["fit", "selection"]).sum()) != manifest["label_budget"]:
        raise ValueError("Participant manifest label budget mismatch")
    if not {"fit", "selection", "dev", "test"} <= set(table["role"]):
        raise ValueError("Participant manifest is missing required roles")
    labels = load_official_labels(config["data"]).table.set_index("participant_id")
    for row in table.itertuples():
        if (
            row.participant_id not in labels.index
            or labels.loc[row.participant_id, "split"] != row.split
        ):
            raise ValueError("Participant manifest disagrees with official split assignments")
        if row.split != "test":
            for column in ("target_binary", "target_phq8"):
                if getattr(row, column) != labels.loc[row.participant_id, column]:
                    raise ValueError("Participant manifest labels changed")
    return manifest


def apply_manifest(dataset, config: dict):
    path = config["experiment"].get("partition_manifest")
    if not path:
        return dataset
    manifest = load_manifest(path, config)
    ids = [
        str(row["participant_id"]) for row in manifest["participants"] if row["role"] != "unused"
    ]
    indexed = dataset.table.set_index("participant_id", drop=False)
    if not set(ids) <= set(indexed.index):
        raise ValueError("Participant manifest requires inputs missing from the prepared dataset")
    table = indexed.loc[ids].reset_index(drop=True)
    return replace(
        dataset,
        table=table,
        partition_manifest=manifest,
        cache_key=f"{dataset.cache_key}_{manifest['manifest_id'][:16]}",
    )


def saved_partition(dataset, training_role="finetune_train", validation_role="finetune_validation"):
    manifest = dataset.partition_manifest
    if manifest is None:
        return None
    from daic_foundation_tab.evaluation.fine_tuning import FineTunePartition

    ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"].tolist()
    positions = {participant_id: position for position, participant_id in enumerate(ids)}
    assignments = pd.DataFrame(manifest["participants"])
    assignments["participant_id"] = assignments["participant_id"].astype(str)
    training = assignments.loc[assignments["role"] == "fit", "participant_id"]
    validation = assignments.loc[assignments["role"] == "selection", "participant_id"]
    assignment = assignments.loc[assignments["role"].isin(["fit", "selection"])].copy()
    assignment["role"] = assignment["role"].map(
        {"fit": training_role, "selection": validation_role}
    )
    target = "target_binary" if manifest["task"] == "classification" else "target_phq8"
    assignment["target"] = assignment[target]
    assignment["random_state"] = manifest["seed"]
    return FineTunePartition(
        np.array([positions[x] for x in training]),
        np.array([positions[x] for x in validation]),
        assignment,
    )
