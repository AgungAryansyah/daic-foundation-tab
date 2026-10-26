from copy import deepcopy

import pytest

from daic_foundation_tab.config import ConfigError, validate_config
from daic_foundation_tab.models.capabilities import model_capabilities


def text_config():
    return {
        "input": {"kind": "transcript"},
        "project": {"seed": 42, "cache_root": "cache", "output_root": "outputs"},
        "runtime": {"require_cuda": True, "device": "cuda:0"},
        "data": {"dataset_id": "synthetic", "transcripts": {"pattern": "{participant_id}.tsv"}},
        "experiment": {"task": "classification", "mode": "frozen_icl"},
        "model": {
            "name": "qwen35_2b",
            "model_id": "Qwen/Qwen3.5-2B",
            "parameters": {"max_length": 4096},
        },
        "evaluation": {"threshold": 0.5, "fine_tuning": {"validation_fraction": 0.2}},
        "tracking": {"wandb": {"enabled": False}},
    }


def test_text_config_does_not_require_aggregation():
    config = text_config()
    validate_config(config)
    config["experiment"].update(task="regression", mode="fine_tuning")
    config["evaluation"]["threshold"] = None
    validate_config(config)
    assert model_capabilities("kumo_medium_ft_regressor").input_kind == "tabular"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "kind", "audio"),
        ("model", "name", "tabiclv2_ft"),
        ("model", "model_id", "other"),
        ("runtime", "device", "cpu"),
    ],
)
def test_rejects_incompatible_configuration(section, key, value):
    config = deepcopy(text_config())
    config[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(config)
