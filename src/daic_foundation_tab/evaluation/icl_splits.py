from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from daic_foundation_tab.data.dataset import ParticipantDataset
from daic_foundation_tab.data.participants import saved_partition
from daic_foundation_tab.data.validation import TrainFeatureSelection, fit_feature_selection
from daic_foundation_tab.evaluation.fine_tuning import (
    FineTunePartition,
    stratified_fine_tune_partition,
)
from daic_foundation_tab.evaluation.regression_splits import regression_fine_tune_partition


@dataclass(frozen=True)
class PreparedICLSplits:
    context_x: pd.DataFrame
    context_y: pd.Series
    reserved_x: pd.DataFrame
    reserved_y: pd.Series
    dev_x: pd.DataFrame
    dev_y: pd.Series
    test_x: pd.DataFrame
    selection: TrainFeatureSelection
    assignment: pd.DataFrame


def prepare_icl_splits(
    dataset: ParticipantDataset,
    feature_set: str,
    task: str,
    validation_fraction: float,
    random_state: int,
) -> PreparedICLSplits:
    target_name = "phq8" if task == "regression" else "binary_depression"
    train_x, train_y = dataset.get_split("train", feature_set, target=target_name)
    dev_x, dev_y = dataset.get_split("dev", feature_set, target=target_name)
    test_x, _ = dataset.get_split("test", feature_set, target=target_name)
    train_ids = dataset.table.loc[dataset.table["split"] == "train", "participant_id"].reset_index(
        drop=True
    )
    target = train_y.reset_index(drop=True)
    partitioner = (
        regression_fine_tune_partition if task == "regression" else stratified_fine_tune_partition
    )
    partition: FineTunePartition = saved_partition(dataset, "icl_context", "reserved_validation") or partitioner(
        target,
        train_ids,
        validation_fraction,
        random_state,
        training_role="icl_context",
        validation_role="reserved_validation",
    )
    selection = fit_feature_selection(train_x.iloc[partition.training_indices])
    columns = selection.columns
    dtype = float if task == "regression" else int
    return PreparedICLSplits(
        context_x=train_x.iloc[partition.training_indices].reindex(columns=columns),
        context_y=target.iloc[partition.training_indices].astype(dtype),
        reserved_x=train_x.iloc[partition.validation_indices].reindex(columns=columns),
        reserved_y=target.iloc[partition.validation_indices].astype(dtype),
        dev_x=dev_x.reindex(columns=columns),
        dev_y=dev_y.astype(dtype),
        test_x=test_x.reindex(columns=columns),
        selection=selection,
        assignment=partition.assignment,
    )
