import pytest
import yaml

from daic_foundation_tab.config import load_config
from daic_foundation_tab.matched_study import condition_configs
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
