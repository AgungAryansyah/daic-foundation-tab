from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import aggregate_source
from .discovery import DiscoveryResult, discover_sources
from .labels import LabelLoadResult, load_official_labels


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class ParticipantDataset:
    table: pd.DataFrame
    manifest: pd.DataFrame
    reconciliation: pd.DataFrame
    inventory: dict[str, Any]
    label_audit: dict[str, Any]
    cache_key: str

    def feature_columns(self, feature_set: str) -> list[str]:
        modalities = {
            "audio": {"audio"},
            "visual": {"visual"},
            "audio_visual": {"audio", "visual"},
        }
        if feature_set not in modalities:
            raise DatasetError(f"Unsupported feature set: {feature_set}")
        return self.manifest.loc[
            self.manifest["modality"].isin(modalities[feature_set]), "column"
        ].tolist()

    def get_split(
        self, split: str, feature_set: str, target: str = "binary_depression"
    ) -> tuple[pd.DataFrame, pd.Series]:
        target_column = {"binary_depression": "target_binary", "phq8": "target_phq8"}.get(target)
        if target_column is None:
            raise DatasetError(f"Unsupported target: {target}")
        subset = self.table.loc[self.table["split"] == split].copy()
        return subset.reindex(columns=self.feature_columns(feature_set)), subset[target_column]


def _enabled_groups(config: dict[str, Any]) -> set[str]:
    groups: set[str] = set()
    for modality_config in config["modalities"].values():
        if modality_config.get("enabled", False):
            groups.update(modality_config.get("groups", []))
    return groups


def _fingerprint(config: dict[str, Any], discovery: DiscoveryResult) -> str:
    sources = []
    for participant in discovery.participants:
        for group, path in participant.files.items():
            if path is not None:
                stat = path.stat()
                sources.append(
                    {
                        "participant_id": participant.participant_id,
                        "group": group,
                        "path": str(path.relative_to(path.parents[2])),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    payload = {
        "source_groups": config["data"]["source_groups"],
        "aggregation": config["aggregation"],
        "sources": sources,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _build_dataset(config: dict[str, Any], labels: LabelLoadResult, discovery: DiscoveryResult) -> ParticipantDataset:
    label_rows = labels.table.set_index("participant_id")
    source_groups = config["data"]["source_groups"]
    required_groups = _enabled_groups(config)
    rows: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    manifest_by_column: dict[str, dict[str, str]] = {}
    source_quality: dict[str, dict[str, Any]] = {}
    exclusions: list[str] = []

    for participant in discovery.participants:
        label = label_rows.loc[participant.participant_id]
        row: dict[str, Any] = {
            "participant_id": participant.participant_id,
            "split": label["split"],
            "target_binary": label["target_binary"],
            "target_phq8": label["target_phq8"],
        }
        reconciliation_row: dict[str, Any] = {
            "participant_id": participant.participant_id,
            "split": label["split"],
            "has_label": bool(pd.notna(label["target_binary"])) or label["split"] == "test",
        }
        missing: list[str] = []

        for group, group_config in source_groups.items():
            path = participant.files[group]
            reconciliation_row[f"has_{group}"] = path is not None
            if group not in required_groups:
                continue
            if path is None:
                missing.append(group)
                continue
            result = aggregate_source(path, group, group_config, config["aggregation"])
            if not result.values:
                missing.append(group)
                continue
            row.update(result.values)
            for item in result.manifest:
                manifest_by_column[item["column"]] = item
            source_quality[f"{participant.participant_id}:{group}"] = result.source_quality

        reconciliation_row["included"] = not missing
        reconciliation_row["exclusion_reason"] = "; ".join(f"missing {group}" for group in missing)
        reconciliation.append(reconciliation_row)
        if missing:
            exclusions.append(
                f"{participant.participant_id} ({label['split']}): {reconciliation_row['exclusion_reason']}"
            )
            continue
        rows.append(row)

    if exclusions and config["data"].get("missing_modality_policy", "error") == "error":
        raise DatasetError("Requested modality unavailable: " + "; ".join(exclusions))

    table = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    manifest = pd.DataFrame(manifest_by_column.values()).sort_values("column").reset_index(drop=True)
    if table["participant_id"].duplicated().any():
        raise DatasetError("Final table contains duplicate participant rows")
    if manifest["column"].duplicated().any():
        raise DatasetError("Feature manifest has duplicate columns")

    inventory = dict(discovery.inventory)
    inventory["source_quality"] = source_quality
    inventory["included_participants"] = len(table)
    inventory["excluded_participants"] = len(exclusions)
    return ParticipantDataset(
        table=table,
        manifest=manifest,
        reconciliation=pd.DataFrame(reconciliation),
        inventory=inventory,
        label_audit=labels.audit,
        cache_key=_fingerprint(config, discovery),
    )


def _cache_paths(config: dict[str, Any], cache_key: str) -> dict[str, Path]:
    root = Path(config["project"]["cache_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"features_{cache_key}"
    return {
        "table": prefix.with_suffix(".parquet"),
        "manifest": root / f"features_{cache_key}_manifest.csv",
        "reconciliation": root / f"features_{cache_key}_reconciliation.csv",
        "inventory": root / f"features_{cache_key}_inventory.json",
        "label_audit": root / f"features_{cache_key}_label_audit.json",
    }


def build_or_load_dataset(config: dict[str, Any]) -> ParticipantDataset:
    labels = load_official_labels(config["data"])
    discovery = discover_sources(labels.table, config["data"])
    cache_key = _fingerprint(config, discovery)
    paths = _cache_paths(config, cache_key)
    if all(path.exists() for path in paths.values()):
        return ParticipantDataset(
            table=pd.read_parquet(paths["table"]),
            manifest=pd.read_csv(paths["manifest"]),
            reconciliation=pd.read_csv(paths["reconciliation"]),
            inventory=json.loads(paths["inventory"].read_text(encoding="utf-8")),
            label_audit=json.loads(paths["label_audit"].read_text(encoding="utf-8")),
            cache_key=cache_key,
        )

    dataset = _build_dataset(config, labels, discovery)
    dataset.table.to_parquet(paths["table"], index=False)
    dataset.manifest.to_csv(paths["manifest"], index=False)
    dataset.reconciliation.to_csv(paths["reconciliation"], index=False)
    paths["inventory"].write_text(json.dumps(dataset.inventory, indent=2), encoding="utf-8")
    paths["label_audit"].write_text(json.dumps(dataset.label_audit, indent=2), encoding="utf-8")
    return dataset
