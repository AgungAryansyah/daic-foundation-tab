from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from daic_foundation_tab.data.dataset import ParticipantDataset
from daic_foundation_tab.data.validation import fit_feature_selection, validate_regression_dataset
from daic_foundation_tab.evaluation.fine_tuning import FineTunePartition, PreparedFineTuneSplits


@dataclass(frozen=True)
class NestedRegressionPartitions:
    outer: FineTunePartition
    inner: FineTunePartition
    assignment: pd.DataFrame


def regression_fine_tune_partition(
    target: pd.Series,
    participant_ids: pd.Series,
    validation_fraction: float,
    random_state: int,
    *,
    training_role: str = "finetune_train",
    validation_role: str = "finetune_validation",
) -> FineTunePartition:
    if len(target) != len(participant_ids) or participant_ids.astype(str).duplicated().any():
        raise ValueError(
            "Regression partition requires unique participant IDs aligned with targets"
        )
    if not np.isfinite(pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)).all():
        raise ValueError("Regression partition requires finite PHQ-8 targets")
    try:
        splitter = ShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=random_state
        )
        training_indices, validation_indices = next(splitter.split(np.zeros(len(target))))
    except ValueError as error:
        raise ValueError("Unable to create a regression fine-tuning validation split") from error
    ordered_indices = np.r_[training_indices, validation_indices]
    assignment = pd.DataFrame(
        {
            "participant_id": participant_ids.iloc[ordered_indices].to_numpy(),
            "role": [training_role] * len(training_indices)
            + [validation_role] * len(validation_indices),
            "target": target.iloc[ordered_indices].to_numpy(dtype=float),
            "random_state": random_state,
        }
    )
    return FineTunePartition(training_indices, validation_indices, assignment)


def nested_regression_partitions(
    target: pd.Series,
    participant_ids: pd.Series,
    outer_validation_fraction: float,
    inner_validation_fraction: float,
    random_state: int,
) -> NestedRegressionPartitions:
    outer = regression_fine_tune_partition(
        target,
        participant_ids,
        outer_validation_fraction,
        random_state,
        training_role="outer_train",
        validation_role="evaluation",
    )
    inner = regression_fine_tune_partition(
        target.iloc[outer.training_indices].reset_index(drop=True),
        participant_ids.iloc[outer.training_indices].reset_index(drop=True),
        inner_validation_fraction,
        random_state + 10_000,
    )
    evaluation = outer.assignment.loc[outer.assignment["role"] == "evaluation"]
    assignment = pd.concat([inner.assignment, evaluation], ignore_index=True)
    return NestedRegressionPartitions(outer, inner, assignment)


def prepare_regression_splits(
    dataset: ParticipantDataset,
    feature_set: str,
    validation_fraction: float,
    random_state: int,
) -> PreparedFineTuneSplits:
    validate_regression_dataset(dataset, feature_set)
    train_x, train_y = dataset.get_split("train", feature_set, target="phq8")
    dev_x, dev_y = dataset.get_split("dev", feature_set, target="phq8")
    test_x, _ = dataset.get_split("test", feature_set, target="phq8")
    train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"].reset_index(
        drop=True
    )
    partition = regression_fine_tune_partition(
        train_y.reset_index(drop=True), train_ids, validation_fraction, random_state
    )
    selection = fit_feature_selection(train_x.iloc[partition.training_indices])
    return PreparedFineTuneSplits(
        training_x=train_x.iloc[partition.training_indices].reindex(columns=selection.columns),
        training_y=train_y.iloc[partition.training_indices].astype(float),
        validation_x=train_x.iloc[partition.validation_indices].reindex(columns=selection.columns),
        validation_y=train_y.iloc[partition.validation_indices].astype(float),
        dev_x=dev_x.reindex(columns=selection.columns),
        dev_y=dev_y.astype(float),
        test_x=test_x.reindex(columns=selection.columns),
        selection=selection,
        assignment=partition.assignment,
    )
