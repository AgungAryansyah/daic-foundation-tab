from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class LabelLoadResult:
    table: pd.DataFrame
    audit: dict[str, Any]


def _normalise_identifier(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _column_name(columns: list[str], candidates: set[str]) -> str | None:
    for column in columns:
        normalised = column.lower().replace("_", "")
        if normalised in candidates:
            return column
    return None


def _read_split(path: Path, split: str) -> tuple[pd.DataFrame, int]:
    if not path.is_file():
        raise LabelError(f"Missing official {split} split file: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    participant_column = _column_name(frame.columns.tolist(), {"participantid", "subjectid", "id"})
    if participant_column is None:
        raise LabelError(f"No participant identifier column in {path}")

    result = pd.DataFrame(
        {
            "participant_id": frame[participant_column].map(_normalise_identifier),
            "split": split,
            "target_binary": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "target_phq8": pd.Series(float("nan"), index=frame.index, dtype="float64"),
        }
    )
    if split == "test":
        return result, 0

    score_column = _column_name(frame.columns.tolist(), {"phq8score"})
    binary_column = _column_name(frame.columns.tolist(), {"phq8binary"})
    if score_column is None:
        raise LabelError(f"No PHQ-8 score column in {path}")

    scores = pd.to_numeric(frame[score_column], errors="coerce")
    if scores.isna().any():
        raise LabelError(f"Non-numeric PHQ-8 score in {path}")
    derived = (scores >= 10).astype("int64")
    result["target_phq8"] = scores
    result["target_binary"] = derived.astype("Int64")

    mismatches = 0
    if binary_column is not None:
        supplied = pd.to_numeric(frame[binary_column], errors="coerce")
        mismatches = int((supplied.notna() & (supplied.astype("Int64") != derived)).sum())
    return result, mismatches


def load_official_labels(data_config: dict[str, Any]) -> LabelLoadResult:
    root = Path(data_config["root"]).expanduser().resolve()
    label_paths = data_config["labels"]
    tables: list[pd.DataFrame] = []
    mismatches: dict[str, int] = {}

    for split in ("train", "dev", "test"):
        table, split_mismatches = _read_split(root / label_paths[split], split)
        tables.append(table)
        mismatches[split] = split_mismatches

    combined = pd.concat(tables, ignore_index=True)
    if combined["participant_id"].duplicated().any():
        duplicate_ids = combined.loc[combined["participant_id"].duplicated(), "participant_id"].tolist()
        raise LabelError(f"Participant overlap across official splits: {duplicate_ids}")

    expected = data_config.get("expected_counts", {})
    actual = combined.groupby("split")["participant_id"].nunique().to_dict()
    warnings = [
        f"expected {count} {split} participants, found {actual.get(split, 0)}"
        for split, count in expected.items()
        if actual.get(split, 0) != count
    ]
    audit = {
        "participant_counts": actual,
        "binary_label_mismatches": mismatches,
        "warnings": warnings,
    }
    return LabelLoadResult(table=combined, audit=audit)
