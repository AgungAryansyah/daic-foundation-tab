import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from daic_foundation_tab.config import load_config
from daic_foundation_tab.data.dataset import build_or_load_dataset
from daic_foundation_tab.data.transcripts import load_transcripts
from daic_foundation_tab.evaluation.fine_tuning import prepare_fine_tune_splits
from daic_foundation_tab.evaluation.icl_splits import prepare_icl_splits
from daic_foundation_tab.matched_study import (
    condition_configs,
    prepare_study,
    verify_prepared_study,
)
from tests.test_participants import manifest_config

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/studies/daic_woz_matched.yaml"


def study_config(tmp_path):
    config = load_config(PRESET, _validate=False)
    data = manifest_config(tmp_path)["data"]
    config["data"].update(data)
    config["data"]["transcripts"]["delimiter"] = ","
    config["evaluation"]["fine_tuning"]["validation_fraction"] = 0.4
    config["tracking"]["wandb"].update(enabled=False, mode="disabled")
    config["project"].update(cache_root=str(tmp_path / "cache"), output_root=str(tmp_path / "runs"))
    config["matched_study"].update(
        tabular_profiles=str(ROOT / "configs/models"),
        text_profile=str(ROOT / "configs/examples/qwen35_2b/base.yaml"),
    )
    config.pop("_config_path", None)
    for participant_id in ["300", "301", "302", "303", "304", "305", "400", "401", "500", "501"]:
        (tmp_path / "data/data" / f"{participant_id}_TRANSCRIPT.csv").write_text(
            "speaker,value\nParticipant,I feel tired\n"
        )
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_preset_defines_all_model_task_adaptation_conditions_without_budgets():
    conditions = condition_configs(PRESET)
    assert len(conditions) == len({name for name, _ in conditions}) == 16
    for _, config in conditions:
        assert "label_budget" not in str(config)
        assert config["study"]["required_inputs"] == ["tabular", "transcript"]
        assert not config["bootstrap"]["enabled"]
        assert not config["evaluation"]["repeated_holdout"]["enabled"]
        assert config["tracking"]["wandb"]["enabled"]
        if config["input"]["kind"] == "transcript":
            assert config["model"]["parameters"]["demonstrations"] == (
                2 if config["experiment"]["mode"] == "frozen_icl" else 0
            )


def test_preparation_matches_participants_and_keeps_separate_task_manifests(tmp_path):
    path = study_config(tmp_path)
    destination = prepare_study(path, tmp_path / "prepared")
    conditions = verify_prepared_study(destination)
    assert len(conditions) == 16
    grouped = {}
    for _, config in conditions:
        task = config["experiment"]["task"]
        grouped.setdefault(task, set()).add(config["tracking"]["wandb"]["group"])
    assert all(len(groups) == 1 for groups in grouped.values())
    assert grouped["classification"] != grouped["regression"]
    config = conditions[0][1]
    dataset = build_or_load_dataset(config)
    ft = prepare_fine_tune_splits(dataset, "audio_visual", 0.2, 999)
    icl = prepare_icl_splits(dataset, "audio_visual", "classification", 0.2, 888)
    assert ft.training_x.equals(icl.context_x)
    text = next(config for _, config in conditions if config["input"]["kind"] == "transcript")
    transcripts = load_transcripts(text)
    assert transcripts.manifest == dataset.partition_manifest
    assert len(transcripts.examples("fit")) == len(ft.training_x)
    plan = json.loads((destination / "campaign.json").read_text())
    assert len(plan["manifests"]) == 2
    assert prepare_study(path, destination) == destination
    assert not (destination / "runs").exists()


def test_preparation_rejects_changed_settings_or_manifest(tmp_path):
    path = study_config(tmp_path)
    destination = prepare_study(path, tmp_path / "prepared")
    config = yaml.safe_load(path.read_text())
    changed = deepcopy(config)
    changed["project"]["seed"] = 123
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="settings changed"):
        prepare_study(path, destination)
    path.write_text(yaml.safe_dump(config))
    manifest_path = destination / "participants_classification.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["participants"][0]["role"] = "test"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        verify_prepared_study(destination)


def test_preparation_rejects_too_many_missing_transcripts(tmp_path):
    path = study_config(tmp_path)
    (tmp_path / "data/data/305_TRANSCRIPT.csv").unlink()
    with pytest.raises(ValueError, match="max_exclusion_fraction"):
        prepare_study(path, tmp_path / "prepared")
