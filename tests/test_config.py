from __future__ import annotations

from pathlib import Path

import pytest

from daic_foundation_tab.config import ConfigError, load_config, validate_config


def test_composes_experiment_and_model_presets() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_visual_complete_cohort.yaml")

    assert config["experiment"]["feature_set"] == "audio_visual"
    assert config["model"]["name"] == "tabiclv2_ft"
    assert config["model"]["parameters"]["epochs"] == 50


def test_rejects_retired_icl_model_name() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["model"]["name"] = "tabiclv2"

    with pytest.raises(ConfigError, match="retired"):
        validate_config(config)


def test_rejects_fine_tuned_test_predictions() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["evaluation"]["test_predictions"] = True

    with pytest.raises(ConfigError, match="finalization"):
        validate_config(config)


def test_rejects_cpu_runtime() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["runtime"]["device"] = "cpu"

    with pytest.raises(ConfigError, match="runtime.device"):
        validate_config(config)


def test_rejects_automatic_fine_tuning_device() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["model"]["parameters"]["device"] = "auto"

    with pytest.raises(ConfigError, match="must match runtime.device"):
        validate_config(config)


def test_rejects_invalid_wandb_mode() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["tracking"]["wandb"]["mode"] = "shared"

    with pytest.raises(ConfigError, match="online, offline, or disabled"):
        validate_config(config)


def test_rejects_malformed_wandb_configuration() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs/experiments/tabiclv2_ft_audio_complete_cohort.yaml")
    config["tracking"]["wandb"] = []

    with pytest.raises(ConfigError, match="must be a mapping"):
        validate_config(config)
