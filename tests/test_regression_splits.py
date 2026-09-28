from __future__ import annotations

import numpy as np
import pandas as pd

from daic_foundation_tab.data.dataset import ParticipantDataset
from daic_foundation_tab.evaluation.regression_splits import (
    nested_regression_partitions,
    prepare_regression_splits,
    regression_fine_tune_partition,
)


def test_regression_partitions_are_reproducible_and_nested() -> None:
    target = pd.Series(np.arange(20, dtype=float))
    participant_ids = pd.Series([str(index) for index in range(len(target))])

    first = nested_regression_partitions(target, participant_ids, 0.2, 0.25, 42)
    second = nested_regression_partitions(target, participant_ids, 0.2, 0.25, 42)

    assert first.assignment.equals(second.assignment)
    assert not set(first.outer.training_indices) & set(first.outer.validation_indices)
    assert not set(first.inner.training_indices) & set(first.inner.validation_indices)
    groups = {
        role: set(group["participant_id"]) for role, group in first.assignment.groupby("role")
    }
    assert not groups["finetune_train"] & groups["finetune_validation"]
    assert not groups["finetune_train"] & groups["evaluation"]
    assert not groups["finetune_validation"] & groups["evaluation"]
    assert set(first.assignment["participant_id"]) == set(participant_ids)


def test_regression_feature_selection_uses_only_training_partition() -> None:
    target = pd.Series(np.arange(20, dtype=float))
    participant_ids = pd.Series([str(index) for index in range(len(target))])
    partition = regression_fine_tune_partition(target, participant_ids, 0.2, 42)
    inner_constant = np.zeros(len(target), dtype=float)
    inner_constant[partition.validation_indices] = 1.0
    table = pd.DataFrame(
        {
            "participant_id": [*participant_ids, "dev_0", "dev_1", "test_0"],
            "split": ["train"] * len(target) + ["dev", "dev", "test"],
            "target_binary": [*(target >= 10).astype(int), 0, 1, pd.NA],
            "target_phq8": [*target, 2.0, 14.0, np.nan],
            "signal": [*target, 20.0, 21.0, 22.0],
            "inner_constant": [*inner_constant, 99.0, 100.0, 101.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "column": ["signal", "inner_constant"],
            "modality": ["audio", "audio"],
            "source_group": ["test", "test"],
            "source_feature": ["signal", "inner_constant"],
            "aggregation": ["mean", "mean"],
        }
    )
    dataset = ParticipantDataset(table, manifest, pd.DataFrame(), {}, {}, "test")

    prepared = prepare_regression_splits(dataset, "audio", 0.2, 42)

    assert prepared.selection.columns == ["signal"]
    assert prepared.dev_y.tolist() == [2.0, 14.0]
    assert prepared.assignment.equals(partition.assignment)
    assert prepared.test_x.columns.tolist() == ["signal"]
