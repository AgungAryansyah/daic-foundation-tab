import pytest
import yaml

from daic_foundation_tab.config import load_config
from daic_foundation_tab.matched_study import condition_configs
from daic_foundation_tab.models.registry import create_text_model
from tests.test_matched_study import ROOT, study_config


@pytest.mark.parametrize("dataset_id", ["edaic", "daic_woz"])
def test_smaller_model_presets_retain_campaign_settings(dataset_id):
    from daic_foundation_tab.transcript_study.bundle import load_settings

    original = load_settings(ROOT / f"configs/studies/{dataset_id}_transcript_cv.yaml")
    small = load_settings(ROOT / f"configs/studies/{dataset_id}_transcript_cv_0_8b.yaml")
    assert small["data"] == original["data"]
    assert small["model"]["parameters"] == original["model"]["parameters"]
    assert small["model"]["name"] == "qwen35_0_8b"
    assert small["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert small["transcript_study"]["model_seeds"] == [42]
    assert small["transcript_study"]["name"] != original["transcript_study"]["name"]


def test_matched_launcher_names_conditions_for_selected_text_model(tmp_path):
    path = study_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["matched_study"]["text_profile"] = str(ROOT / "configs/examples/qwen35_0_8b/base.yaml")
    path.write_text(yaml.safe_dump(config))
    conditions = condition_configs(path)
    text = [(name, cfg) for name, cfg in conditions if cfg["input"]["kind"] == "transcript"]
    assert len(conditions) == 16 and len(text) == 4
    assert all("qwen35_0_8b" in name for name, _ in text)
    assert all(cfg["model"]["model_id"] == "Qwen/Qwen3.5-0.8B" for _, cfg in text)


@pytest.mark.parametrize("name", ["base", "classification_icl", "classification_lora", "score_icl", "score_lora"])
def test_smaller_standalone_examples_validate(name):
    config = load_config(ROOT / f"configs/examples/qwen35_0_8b/{name}.yaml")
    assert config["model"]["name"] == "qwen35_0_8b"
    assert config["model"]["model_id"] == "Qwen/Qwen3.5-0.8B"


@pytest.mark.parametrize("dataset_id", ["edaic", "daic_woz"])
@pytest.mark.parametrize("size", ["0_8b", "2b", "9b"])
def test_dora_campaign_presets(dataset_id, size):
    from daic_foundation_tab.transcript_study.bundle import load_settings

    base = ROOT / "configs/studies"
    config = load_settings(base / f"{dataset_id}_transcript_cv_dora_{size}.yaml")
    original = load_settings(base / f"{dataset_id}_transcript_cv.yaml")
    assert config["data"] == original["data"]
    assert config["model"]["parameters"] == {**original["model"]["parameters"], "adapter_method": "dora"}
    assert config["transcript_study"]["methods"] == (
        ["frozen_icl", "fine_tuning"] if size == "9b" else ["fine_tuning"]
    )
    assert config["model"]["name"] == f"qwen35_{size}"
    if size == "9b":
        assert config["model"]["revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert create_text_model(config).adapter_method == "dora"
