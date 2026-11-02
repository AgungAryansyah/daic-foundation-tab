import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from daic_foundation_tab.tabular_study.bundle import prepare_bundle as prepare_tabular
from daic_foundation_tab.transcript_study.bundle import (
    dataset_for,
    prepare_bundle,
    read_bundle,
    roles_for,
)
from tests.test_tabular_study_bundle import study_input
from tests.test_text_prompt import TinyTokenizer

ROOT = Path(__file__).parents[1]


def transcript_input(tmp_path, dataset_id="edaic"):
    source = study_input(tmp_path)
    raw = yaml.safe_load(source.read_text())
    raw["data"]["dataset_id"] = dataset_id
    source.write_text(yaml.safe_dump(raw))
    parent = prepare_tabular(source, tmp_path / "tabular")
    table = pd.read_parquet(parent / "features.parquet")
    root = Path(raw["data"]["root"])
    for participant in table["participant_id"].astype(str):
        if dataset_id == "edaic":
            folder = root / "data" / f"{participant}_P"
            folder.mkdir()
            (folder / f"{participant}_Transcript.csv").write_text(
                "Start_Time,End_Time,Text,Confidence\n1,2,I feel tired.,0.9\n"
            )
        else:
            (root / "data" / f"{participant}_TRANSCRIPT.csv").write_text(
                "speaker\tvalue\nEllie\tHow are you?\nParticipant\tI feel tired.\n"
            )
    config = tmp_path / "transcript.yaml"
    config.write_text(yaml.safe_dump({
        "extends": str(ROOT / f"configs/studies/{dataset_id}_transcript_cv.yaml"),
        "data": {"root": str(root), "dataset_id": dataset_id},
        "tracking": {"wandb": {"enabled": False, "mode": "disabled"}},
        "transcript_study": {"bootstrap_iterations": 5},
    }))
    return config, parent, root


@pytest.mark.parametrize("dataset_id", ["edaic", "daic_woz"])
def test_transcript_cohort_and_folds_are_exact_and_portable(tmp_path, dataset_id):
    config, parent, _ = transcript_input(tmp_path, dataset_id)
    original = {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()}
    bundle = prepare_bundle(config, parent, tmp_path / "transcript", tokenizer=TinyTokenizer())
    study, rows, texts, folds = context = read_bundle(bundle)
    assert len(study["conditions"]["cv"]) == 20
    assert len(study["conditions"]["final"]) == 4
    assert study["configuration"]["data"]["transcripts"]["format"] == dataset_id
    assert json.loads((parent / "folds.json").read_text()) == folds
    assert all(text == "I feel tired." for text in texts.values())
    assert all(row["target_phq8"] is None for row in rows if row["split"] == "test")
    for condition in study["conditions"]["cv"]:
        roles = roles_for(rows, folds, condition)
        dataset = dataset_for(context, condition)
        assert {x.participant_id for x in dataset.examples("evaluation")} == set(roles["evaluation"])
        assert all(x.target is None for x in dataset.examples("evaluation"))
    relocated = tmp_path / "other-host"
    shutil.copytree(bundle, relocated)
    assert read_bundle(relocated)[0] == study
    assert original == {p.name: p.read_bytes() for p in parent.iterdir() if p.is_file()}
    again = prepare_bundle(config, parent, tmp_path / "again", tokenizer=TinyTokenizer())
    assert read_bundle(again)[0] == study


@pytest.mark.parametrize("problem", ["missing", "duplicate", "empty"])
def test_transcript_preparation_rejects_incomplete_sources(tmp_path, problem):
    config, parent, root = transcript_input(tmp_path)
    path = next((root / "data").rglob("*_Transcript.csv"))
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        raw = yaml.safe_load(config.read_text())
        raw["data"]["transcripts"] = {
            "pattern": "data/{participant_id}_P/{participant_id}_Transcript*.csv"
        }
        config.write_text(yaml.safe_dump(raw))
        shutil.copyfile(path, path.with_name(path.stem + "_duplicate.csv"))
    else:
        path.write_text("Start_Time,End_Time,Text,Confidence\n1,2,,0.9\n")
    with pytest.raises(ValueError, match="exactly one|Empty selected"):
        prepare_bundle(config, parent, tmp_path / "transcript", tokenizer=TinyTokenizer())


def test_changed_transcripts_invalidate_identity_and_bundle(tmp_path):
    config, parent, root = transcript_input(tmp_path)
    bundle = prepare_bundle(config, parent, tmp_path / "transcript", tokenizer=TinyTokenizer())
    old = read_bundle(bundle)[0]["study_id"]
    next((root / "data").rglob("*_Transcript.csv")).write_text(
        "Start_Time,End_Time,Text,Confidence\n1,2,Different speech.,0.9\n"
    )
    other = prepare_bundle(config, parent, tmp_path / "different", tokenizer=TinyTokenizer())
    assert read_bundle(other)[0]["study_id"] != old
    (bundle / "transcripts.json").write_text("{}")
    with pytest.raises(ValueError, match="content changed"):
        read_bundle(bundle)
