from __future__ import annotations

from pathlib import Path

from daic_foundation_tab.config import load_config


def test_composes_experiment_and_model_presets() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_audio_visual.yaml")

    assert config["experiment"]["feature_set"] == "audio_visual"
    assert config["model"]["name"] == "tabiclv2"
    assert config["model"]["parameters"]["n_estimators"] == 8
