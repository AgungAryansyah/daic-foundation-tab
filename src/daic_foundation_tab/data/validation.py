from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .dataset import DatasetError, ParticipantDataset

EXACT_DENYLIST = {
    "phq",
    "phq8",
    "phq_8",
    "score",
    "target",
    "label",
    "depressed",
    "depression_label",
    "participant_id",
    "subject_id",
    "id",
}


@dataclass(frozen=True)
class TrainFeatureSelection:
    columns: list[str]
    dropped_all_nan: list[str]
    dropped_constant: list[str]
    near_constant: list[str]


@dataclass(frozen=True)
class PreparedSplits:
    train_x: pd.DataFrame
    train_y: pd.Series
    dev_x: pd.DataFrame
    dev_y: pd.Series
    test_x: pd.DataFrame
    selection: TrainFeatureSelection


def _offending_columns(columns: list[str]) -> list[str]:
    offending = []
    for column in columns:
        normalised = column.lower()
        tokens = set(normalised.replace("-", "_").split("_"))
        if normalised in EXACT_DENYLIST or bool(tokens & EXACT_DENYLIST):
            offending.append(column)
    return offending


def fit_feature_selection(train_x: pd.DataFrame) -> TrainFeatureSelection:
    dropped_all_nan = [column for column in train_x if train_x[column].isna().all()]
    dropped_constant = [
        column
        for column in train_x.columns.difference(dropped_all_nan)
        if train_x[column].notna().all() and train_x[column].nunique(dropna=False) == 1
    ]
    candidate_columns = [
        column for column in train_x if column not in set(dropped_all_nan) | set(dropped_constant)
    ]
    near_constant = [
        column
        for column in candidate_columns
        if train_x[column].notna().any()
        and train_x[column].dropna().value_counts(normalize=True).iloc[0] >= 0.95
    ]
    return TrainFeatureSelection(
        columns=candidate_columns,
        dropped_all_nan=dropped_all_nan,
        dropped_constant=dropped_constant,
        near_constant=near_constant,
    )


def validate_dataset(
    dataset: ParticipantDataset, feature_set: str, max_exclusion_fraction: float = 0.05
) -> dict[str, Any]:
    if dataset.table["participant_id"].duplicated().any():
        raise DatasetError("Final table contains duplicate participant rows")
    if dataset.manifest["column"].duplicated().any():
        raise DatasetError("Feature manifest contains duplicate columns")
    if dataset.manifest["modality"].isna().any() or dataset.manifest["source_group"].isna().any():
        raise DatasetError("Every feature must have modality and source-group provenance")

    feature_columns = dataset.feature_columns(feature_set)
    offending = _offending_columns(feature_columns)
    if offending:
        raise DatasetError(f"Target or identifier leakage detected: {offending}")

    split_report: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        subset = dataset.table.loc[dataset.table["split"] == split]
        target = subset["target_binary"].dropna().astype(int)
        split_report[split] = {
            "participants": len(subset),
            "depressed": int(target.sum()),
            "non_depressed": int((target == 0).sum()),
            "positive_rate": float(target.mean()) if not target.empty else None,
            "feature_count": len(feature_columns),
            "missing_cell_percentage": float(subset[feature_columns].isna().to_numpy().mean() * 100),
            "complete_cases": int(subset[feature_columns].notna().all(axis=1).sum()),
        }
    train_x, _ = dataset.get_split("train", feature_set)
    selection = fit_feature_selection(train_x)
    exclusion_report: dict[str, Any] = {}
    exclusion_warnings: list[str] = []
    reconciliation_columns = {"split", "included", "target_binary"}
    if reconciliation_columns.issubset(dataset.reconciliation.columns):
        for split, reconciliation in dataset.reconciliation.groupby("split", dropna=False):
            excluded = reconciliation.loc[~reconciliation["included"]]
            fraction = float(len(excluded) / len(reconciliation)) if len(reconciliation) else 0.0
            exclusion_report[str(split)] = {
                "participants": len(reconciliation),
                "excluded": len(excluded),
                "exclusion_fraction": fraction,
                "excluded_depressed": int(excluded["target_binary"].eq(1).sum()),
                "excluded_non_depressed": int(excluded["target_binary"].eq(0).sum()),
            }
            if fraction > max_exclusion_fraction:
                exclusion_warnings.append(
                    f"{split} exclusion fraction {fraction:.3f} exceeds {max_exclusion_fraction:.3f}"
                )
    return {
        "feature_set": feature_set,
        "split_statistics": split_report,
        "feature_count_before_filtering": len(feature_columns),
        "dropped_all_nan": selection.dropped_all_nan,
        "dropped_constant": selection.dropped_constant,
        "near_constant": selection.near_constant,
        "label_audit": dataset.label_audit,
        "exclusions": exclusion_report,
        "warnings": exclusion_warnings,
    }


def prepare_splits(dataset: ParticipantDataset, feature_set: str) -> PreparedSplits:
    validate_dataset(dataset, feature_set)
    train_x, train_y = dataset.get_split("train", feature_set)
    dev_x, dev_y = dataset.get_split("dev", feature_set)
    test_x, _ = dataset.get_split("test", feature_set)
    selection = fit_feature_selection(train_x)
    columns = selection.columns
    return PreparedSplits(
        train_x=train_x.reindex(columns=columns),
        train_y=train_y.astype(int),
        dev_x=dev_x.reindex(columns=columns),
        dev_y=dev_y.astype(int),
        test_x=test_x.reindex(columns=columns),
        selection=selection,
    )
