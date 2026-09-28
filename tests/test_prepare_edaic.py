from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from daic_foundation_tab.config import load_config
from daic_foundation_tab.data.dataset import build_or_load_dataset
from daic_foundation_tab.data.labels import load_test_ground_truth
from daic_foundation_tab.data.prepare_edaic import PreparationError, prepare_edaic

COUNTS = {"train": 1, "dev": 1, "test": 1}


def _raw_edaic(root: Path) -> None:
    (root / "labels").mkdir(parents=True)
    for split, participant_id, score, binary in (
        ("train", "300", 12, 0),
        ("dev", "301", 3, 0),
        ("test", "302", 11, 0),
    ):
        pd.DataFrame(
            {"Participant_ID": [participant_id], "PHQ_Score": [score], "PHQ_Binary": [binary]}
        ).to_csv(root / "labels" / f"{split}_split.csv", index=False)
        features = root / "data" / f"{participant_id}_P" / "features"
        features.mkdir(parents=True)
        (features / f"{participant_id}_OpenSMILE2.3.0_egemaps.csv").write_text(
            "name;frameTime;Loudness_sma3\nclip;0.0;1.0\nclip;0.1;3.0\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "frame": [0, 1],
                "timestamp": [0.0, 0.1],
                "success": [1, 1],
                "pose_Tx": [2.0, 4.0],
            }
        ).to_csv(features / f"{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv", index=False)


def test_prepare_edaic_makes_score_based_labels_and_numeric_features(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    _raw_edaic(source)

    status = prepare_edaic(source, output, COUNTS)
    prepare_edaic(source, output, COUNTS)

    assert status["supplied_binary_disagreements"] == {"train": 1, "dev": 0, "test": 1}
    with (output / "original_labels" / "train_split.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"Participant_ID": "300", "PHQ8_Score": "12"}]
    assert json.loads((output / "preparation_status.json").read_text())["phase"] == "complete"

    config = load_config("configs/experiments/tabiclv2_ft_edaic_audio_visual.yaml")
    config["data"]["root"] = str(output)
    config["data"]["expected_counts"] = COUNTS
    config["project"]["cache_root"] = str(tmp_path / "cache")
    dataset = build_or_load_dataset(config)

    assert len(dataset.table) == 3
    assert set(dataset.manifest["source_group"]) == {"opensmile_egemaps", "openface_21"}
    assert "opensmile_egemaps_frameTime_mean" not in dataset.table
    assert dataset.table.loc[dataset.table["participant_id"] == "300", "target_binary"].item() == 1
    assert load_test_ground_truth(
        output / "original_labels" / "test_split.csv", pd.Series(["302"])
    ).tolist() == [1]


def test_prepare_edaic_reports_missing_feature_before_copying(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "prepared"
    _raw_edaic(source)
    (source / "data" / "302_P" / "features" / "302_OpenFace2.1.0_Pose_gaze_AUs.csv").unlink()

    with pytest.raises(PreparationError, match="Missing or empty feature file"):
        prepare_edaic(source, output, COUNTS)

    status = json.loads((output / "preparation_status.json").read_text())
    assert status["phase"] == "failed"
    assert status["completed_participants"] == 2
    assert not (output / "data").exists()
