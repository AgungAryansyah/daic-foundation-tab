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


@pytest.mark.parametrize("score_column", ["PHQ8_Score", "PHQ_Score"])
def test_official_labels_preserve_phq8_scores_and_hide_test_scores(tmp_path, score_column) -> None:
    config = write_synthetic_dataset(tmp_path / "dataset")
    label_root = tmp_path / "dataset" / "original_labels"
    for name in ("train.csv", "dev.csv"):
        path = label_root / name
        frame = pd.read_csv(path).rename(columns={"PHQ8_Score": score_column})
        frame.to_csv(path, index=False)

    labels = load_official_labels(config["data"]).table

    assert labels.loc[labels["participant_id"] == "300", "target_phq8"].iloc[0] == 2.0
    assert labels.loc[labels["participant_id"] == "400", "target_phq8"].iloc[0] == 5.0
    assert labels.loc[labels["split"] == "test", "target_phq8"].isna().all()


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


@pytest.mark.parametrize("score_column", ["PHQ8_Score", "PHQ_Score"])
def test_phq8_ground_truth_uses_score_and_requested_order(tmp_path, score_column) -> None:
    path = tmp_path / "full_test_split.csv"
    pd.DataFrame({"Participant_ID": [500, 501], score_column: [3, 14]}).to_csv(path, index=False)

    scores = load_test_ground_truth(path, pd.Series(["501", "500"]), target="phq8")
    binary = load_test_ground_truth(path, pd.Series(["501", "500"]))

    assert scores.tolist() == [14.0, 3.0]
    assert binary.tolist() == [1, 0]


def test_phq8_ground_truth_rejects_inconsistent_binary_label(tmp_path) -> None:
    path = tmp_path / "full_test_split.csv"
    pd.DataFrame({"Participant_ID": [500], "PHQ8_Score": [12], "PHQ8_Binary": [0]}).to_csv(
        path, index=False
    )

    with pytest.raises(LabelError, match="disagree"):
        load_test_ground_truth(path, pd.Series(["500"]), target="phq8")
