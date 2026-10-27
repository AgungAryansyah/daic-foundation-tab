import json
import shutil
from copy import deepcopy

import pandas as pd
import pytest
import yaml

from daic_foundation_tab.tabular_study.bundle import prepare_bundle, read_bundle, verify_bundle
from tests.helpers import write_synthetic_dataset
from tests.test_tabular_study_configuration import PRESET, ROOT


def study_input(tmp_path):
    root = tmp_path / "data"
    base = write_synthetic_dataset(root)
    train = pd.read_csv(root / "original_labels/train.csv")
    for index in range(306, 330):
        template = 300 + index % 6
        for path in (root / "data").glob(f"{template}_*"):
            shutil.copyfile(path, root / "data" / path.name.replace(str(template), str(index), 1))
        score = 4 if index % 2 == 0 else 14
        train.loc[len(train)] = [index, int(score >= 10), score]
    train.to_csv(root / "original_labels/train.csv", index=False)
    base["data"]["dataset_id"] = "synthetic"
    config = {
        "extends": str(PRESET),
        "project": base["project"],
        "data": base["data"],
        "modalities": base["modalities"],
        "tracking": base["tracking"],
        "tabular_study": {
            "tabular_profiles": str(ROOT / "configs/models"),
            "bootstrap_iterations": 10,
        },
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_bundle_is_portable_and_reuses_data_for_new_model_seeds(tmp_path):
    config = study_input(tmp_path)
    output = prepare_bundle(config, tmp_path / "prepared")
    study, table, features, folds = read_bundle(output)
    assert len(study["conditions"]["cv"]) == 120
    assert len(table) == 34 and set(features["modality"]) == {"audio", "visual"}
    assert table.loc[table["split"] == "test", ["target_phq8", "target_binary"]].isna().all().all()
    assert len(folds["classification"]["folds"]) == 5
    relocated = tmp_path / "elsewhere"
    shutil.copytree(output, relocated)
    assert verify_bundle(relocated)["study_id"] == study["study_id"]
    changed = deepcopy(yaml.safe_load(config.read_text()))
    changed["tabular_study"]["model_seeds"] = [42, 7]
    config.write_text(yaml.safe_dump(changed))
    extended = prepare_bundle(config, tmp_path / "extended", reuse=relocated)
    expanded = verify_bundle(extended)
    assert expanded["study_id"] == study["study_id"]
    assert len(expanded["conditions"]["cv"]) == 240
    assert {x["condition_id"] for x in study["conditions"]["cv"]} <= {
        x["condition_id"] for x in expanded["conditions"]["cv"]
    }


def test_bundle_detects_modified_data_and_refuses_different_protocol(tmp_path):
    config = study_input(tmp_path)
    output = prepare_bundle(config, tmp_path / "prepared")
    changed = deepcopy(yaml.safe_load(config.read_text()))
    changed["tabular_study"]["fold_seed"] = 3
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="Only model seeds"):
        prepare_bundle(config, tmp_path / "different", reuse=output)
    folds_path = output / "folds.json"
    folds = json.loads(folds_path.read_text())
    folds["classification"]["folds"][0]["fit"].pop()
    folds_path.write_text(json.dumps(folds))
    with pytest.raises(ValueError, match="content changed"):
        verify_bundle(output)
