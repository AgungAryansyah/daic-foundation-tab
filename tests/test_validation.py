from __future__ import annotations

import pandas as pd
import pytest

from daic_foundation_tab.data.dataset import DatasetError, ParticipantDataset
from daic_foundation_tab.data.validation import prepare_splits, validate_dataset


def _dataset(feature_name: str) -> ParticipantDataset:
    return ParticipantDataset(
        table=pd.DataFrame(
            {
                "participant_id": ["1", "2", "3", "4", "5", "6"],
                "split": ["train", "train", "train", "train", "dev", "test"],
                "target_binary": [0, 1, 0, 1, 0, pd.NA],
                "target_phq8": [1.0, 12.0, 2.0, 13.0, 3.0, float("nan")],
                feature_name: [1.0, 1.0, 1.0, 1.0, 99.0, 100.0],
            }
        ),
        manifest=pd.DataFrame(
            [
                {
                    "column": feature_name,
                    "modality": "audio",
                    "source_group": "covarep",
                    "source_feature": "F0",
                    "aggregation": "mean",
                }
            ]
        ),
        reconciliation=pd.DataFrame(),
        inventory={},
        label_audit={},
        cache_key="test",
    )


def test_rejects_target_leakage_from_manifest() -> None:
    with pytest.raises(DatasetError, match="leakage"):
        validate_dataset(_dataset("phq8_score_mean"), "audio")


def test_feature_filtering_uses_train_rows_only() -> None:
    prepared = prepare_splits(_dataset("covarep_F0_mean"), "audio")

    assert prepared.selection.dropped_constant == ["covarep_F0_mean"]
    assert prepared.dev_x.empty
