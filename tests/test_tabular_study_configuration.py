from pathlib import Path

import pytest
import yaml

from daic_foundation_tab.config import ConfigError
from daic_foundation_tab.tabular_study.configuration import condition_specs, load_study_config

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/studies/daic_woz_tabular_cv.yaml"


def test_default_study_resolves_all_conditions_and_profiles():
    config = load_study_config(PRESET)
    cv = condition_specs(config)
    final = condition_specs(config, "final")
    assert len(cv) == 120 and len(final) == 24
    assert len({(x["family"], x["modality"]) for x in cv}) == 6
    assert all(
        sum((x["family"], x["modality"]) == group for x in cv) == 20
        for group in {(x["family"], x["modality"]) for x in cv}
    )
    assert {x["model_seed"] for x in cv} == {42}
    assert all("transcript" not in str(x) for x in cv)
    assert (
        config["profiles"]["tabpfn35:regression:fine_tuning"]["parameters"]["eval_metric"] == "mse"
    )
    assert (
        config["profiles"]["kumo_medium:regression:fine_tuning"]["parameters"]["eval_metric"]
        == "mse"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_seeds", [42, 42]),
        ("folds", 1),
        ("modalities", ["unknown"]),
        ("selection_fraction", 1),
    ],
)
def test_invalid_sweep_settings_are_rejected(tmp_path, key, value):
    override = {
        "extends": str(PRESET),
        "tabular_study": {key: value, "tabular_profiles": str(ROOT / "configs/models")},
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(override))
    with pytest.raises(ConfigError):
        load_study_config(path)
