from __future__ import annotations

import pandas as pd

from daic_foundation_tab.data.labels import load_official_labels

from .helpers import write_synthetic_dataset


def test_score_threshold_is_the_canonical_binary_target(tmp_path) -> None:
    config = write_synthetic_dataset(tmp_path / "dataset")
    train_path = tmp_path / "dataset" / "original_labels" / "train.csv"
    train = pd.read_csv(train_path)
    train.loc[0, "PHQ8_Binary"] = 1
    train.to_csv(train_path, index=False)

    labels = load_official_labels(config["data"])
    first_train = labels.table.loc[labels.table["participant_id"] == "300"].iloc[0]

    assert first_train["target_binary"] == 0
    assert labels.audit["binary_label_mismatches"]["train"] == 1
