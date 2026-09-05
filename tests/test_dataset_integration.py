from __future__ import annotations

from daic_foundation_tab.data.dataset import build_or_load_dataset
from daic_foundation_tab.data.validation import prepare_splits, validate_dataset

from .helpers import write_synthetic_dataset


def test_builds_one_participant_row_and_complete_manifest(tmp_path) -> None:
    config = write_synthetic_dataset(tmp_path / "dataset")
    dataset = build_or_load_dataset(config)
    report = validate_dataset(dataset, "audio_visual")
    prepared = prepare_splits(dataset, "audio_visual")

    assert len(dataset.table) == 10
    assert dataset.table["participant_id"].is_unique
    assert set(dataset.manifest["modality"]) == {"audio", "visual"}
    assert len(prepared.train_x) == 6
    assert len(prepared.dev_x) == 2
    assert len(prepared.test_x) == 2
    assert report["split_statistics"]["train"]["participants"] == 6


def test_explicit_complete_cohort_policy_logs_missing_modalities(tmp_path) -> None:
    config = write_synthetic_dataset(tmp_path / "dataset")
    (tmp_path / "dataset" / "data" / "400_COVAREP.csv").unlink()
    config["data"]["missing_modality_policy"] = "exclude"

    dataset = build_or_load_dataset(config)

    excluded = dataset.reconciliation.loc[dataset.reconciliation["participant_id"] == "400"].iloc[0]
    assert not excluded["included"]
    assert excluded["exclusion_reason"] == "missing covarep"
