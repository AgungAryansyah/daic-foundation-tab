from __future__ import annotations

import pandas as pd
import pytest

from daic_foundation_tab.data.labels import LabelError, load_official_labels, load_test_ground_truth

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


def test_test_ground_truth_rejects_missing_participant(tmp_path) -> None:
    path = tmp_path / "full_test_split.csv"
    pd.DataFrame({"Participant_ID": [500], "PHQ_Score": [2]}).to_csv(path, index=False)

    with pytest.raises(LabelError, match="Missing test ground truth"):
        load_test_ground_truth(path, pd.Series(["500", "501"]))


def test_test_ground_truth_rejects_inconsistent_binary_label(tmp_path) -> None:
    path = tmp_path / "full_test_split.csv"
    pd.DataFrame({"Participant_ID": [500], "PHQ_Score": [12], "PHQ_Binary": [0]}).to_csv(
        path, index=False
    )

    with pytest.raises(LabelError, match="disagree"):
        load_test_ground_truth(path, pd.Series(["500"]))
