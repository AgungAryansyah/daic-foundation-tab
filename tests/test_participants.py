import json

import pytest

from daic_foundation_tab.data.dataset import build_or_load_dataset
from daic_foundation_tab.data.participants import fingerprint, load_manifest, prepare_manifest
from daic_foundation_tab.evaluation.fine_tuning import prepare_fine_tune_splits
from daic_foundation_tab.evaluation.icl_splits import prepare_icl_splits
from tests.helpers import write_synthetic_dataset


def manifest_config(tmp_path):
    config = write_synthetic_dataset(tmp_path / "data")
    config["data"]["dataset_id"] = "synthetic"
    config["evaluation"]["fine_tuning"]["validation_fraction"] = 0.4
    return config


def test_frozen_and_finetuned_use_identical_saved_participants(tmp_path):
    config = manifest_config(tmp_path)
    manifest = prepare_manifest(config)
    path = tmp_path / "participants.json"
    path.write_text(json.dumps(manifest))
    config["experiment"]["partition_manifest"] = str(path)
    dataset = build_or_load_dataset(config)
    ft = prepare_fine_tune_splits(dataset, "audio_visual", 0.2, 999)
    icl = prepare_icl_splits(dataset, "audio_visual", "classification", 0.2, 888)
    assert ft.training_x.equals(icl.context_x)
    assert ft.validation_x.equals(icl.reserved_x)
    assert len(ft.training_y) + len(ft.validation_y) == 6
    assert dataset.partition_manifest == manifest


def test_manifest_rejects_dataset_mismatch_and_split_leakage(tmp_path):
    config = manifest_config(tmp_path)
    manifest = prepare_manifest(config)
    path = tmp_path / "participants.json"
    manifest["participants"][0]["role"] = "test"
    manifest["manifest_id"] = fingerprint({k: v for k, v in manifest.items() if k != "manifest_id"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leak"):
        load_manifest(path, config)
    config["data"]["dataset_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        load_manifest(path, config)


def test_intersection_excludes_missing_transcripts_and_metadata_files(tmp_path):
    config = manifest_config(tmp_path)
    config["data"]["transcripts"] = {"pattern": "data/*{participant_id}_TRANSCRIPT.csv"}
    config["study"] = {"required_inputs": ["tabular", "transcript"]}
    for participant_id in ["300", "301", "302", "303", "304", "400", "401", "500", "501"]:
        (tmp_path / "data/data" / f"{participant_id}_TRANSCRIPT.csv").write_text("speaker,value\n")
    (tmp_path / "data/data/._305_TRANSCRIPT.csv").write_bytes(b"metadata")
    manifest = prepare_manifest(config)
    assert manifest["exclusions"] == [
        {"participant_id": "305", "exclusion_reason": "missing transcript"}
    ]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_full_training_cohort_is_reproducible(tmp_path, task):
    config = manifest_config(tmp_path)
    config["experiment"]["task"] = task
    assert prepare_manifest(config) == prepare_manifest(config)
    manifest = prepare_manifest(config)
    assert sum(row["role"] in {"fit", "selection"} for row in manifest["participants"]) == 6
    assert all(
        row["role"] in {"fit", "selection", "dev", "test"} for row in manifest["participants"]
    )
    assert "label_budget" not in manifest


def test_excluded_participants_do_not_enter_feature_preparation(tmp_path):
    config = manifest_config(tmp_path)
    for path in (tmp_path / "data/data").glob("305_*"):
        path.unlink()
    manifest = prepare_manifest(config)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config["experiment"]["partition_manifest"] = str(path)
    dataset = build_or_load_dataset(config)
    assert "305" not in set(dataset.table["participant_id"])


def test_removed_subsampling_setting_is_rejected(tmp_path):
    config = manifest_config(tmp_path)
    config["study"] = {"label_budget": 4}
    with pytest.raises(ValueError, match="was removed"):
        prepare_manifest(config)


def test_manifest_cannot_silently_omit_eligible_training_participants(tmp_path):
    config = manifest_config(tmp_path)
    manifest = prepare_manifest(config)
    manifest["participants"] = [
        row for row in manifest["participants"] if row["participant_id"] != "300"
    ]
    manifest["manifest_id"] = fingerprint({k: v for k, v in manifest.items() if k != "manifest_id"})
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="full eligible cohort"):
        load_manifest(path, config)


def test_preparation_reports_progress_from_start(tmp_path):
    config = manifest_config(tmp_path)
    calls = []
    prepare_manifest(config, lambda completed, total: calls.append((completed, total)))
    assert calls[0] == (0, 10)
    assert calls[-1] == (10, 10)
