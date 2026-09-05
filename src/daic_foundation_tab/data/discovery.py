from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class DiscoveryError(ValueError):
    pass


@dataclass(frozen=True)
class ParticipantSources:
    participant_id: str
    split: str
    files: dict[str, Path | None]


@dataclass(frozen=True)
class DiscoveryResult:
    participants: list[ParticipantSources]
    inventory: dict[str, Any]


def _matches(root: Path, pattern: str, participant_id: str) -> list[Path]:
    candidate_pattern = pattern.format(participant_id=participant_id)
    return sorted(path for path in root.glob(candidate_pattern) if path.is_file() and not path.name.startswith("."))


def discover_sources(labels: pd.DataFrame, data_config: dict[str, Any]) -> DiscoveryResult:
    root = Path(data_config["root"]).expanduser().resolve()
    if not root.is_dir():
        raise DiscoveryError(f"Dataset root does not exist: {root}")

    source_groups = data_config["source_groups"]
    participants: list[ParticipantSources] = []
    counts = {group: 0 for group in source_groups}

    for row in labels.itertuples(index=False):
        files: dict[str, Path | None] = {}
        for group, group_config in source_groups.items():
            matches = _matches(root, group_config["pattern"], row.participant_id)
            if len(matches) > 1:
                candidates = ", ".join(str(path) for path in matches)
                raise DiscoveryError(
                    f"Ambiguous {group} source for participant {row.participant_id}: {candidates}"
                )
            files[group] = matches[0] if matches else None
            counts[group] += int(bool(matches))
        participants.append(ParticipantSources(row.participant_id, row.split, files))

    inventory = {
        "dataset_root": str(root),
        "participants": len(participants),
        "source_groups": {
            group: {
                "found_participants": count,
                "pattern": source_groups[group]["pattern"],
                "parser": source_groups[group]["parser"],
            }
            for group, count in counts.items()
        },
    }
    return DiscoveryResult(participants=participants, inventory=inventory)
