import json

import pandas as pd
import pytest

from daic_foundation_tab.evaluation.paired import paired_comparison


def write_run(root, name, task="classification"):
    path = root / name
    path.mkdir()
    record = {
        "status": "completed",
        "paired_comparison_verified": True,
        "dataset_id": "synthetic",
        "task": task,
        "manifest_id": "same-manifest",
        "cohort_id": "same-cohort",
        "label_budget": 20,
    }
    (path / "study_run.json").write_text(json.dumps(record))
    table = (
        {
            "participant_id": ["a", "b", "c", "d"],
            "split": ["dev"] * 4,
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 0, 1],
            "prob_depressed": [0.1, 0.9, 0.2, 0.8],
        }
        if task == "classification"
        else {
            "participant_id": ["a", "b", "c", "d"],
            "split": ["dev"] * 4,
            "phq8_true": [1, 12, 2, 14],
            "phq8_pred": [2, 13, 3, 15],
        }
    )
    pd.DataFrame(table).to_csv(path / "predictions_dev.csv", index=False)
    return path


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_paired_bootstrap_uses_the_same_participants(tmp_path, task):
    left, right = write_run(tmp_path, "left", task), write_run(tmp_path, "right", task)
    table = paired_comparison(left, right, iterations=20)
    assert (table["delta_right_minus_left"] == 0).all()
    assert table["ci_lower"].dropna().eq(0).all()
    assert table["ci_upper"].dropna().eq(0).all()


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "failed"),
        ("dataset_id", "other"),
        ("manifest_id", "different"),
        ("paired_comparison_verified", False),
    ],
)
def test_incompatible_runs_are_rejected(tmp_path, key, value):
    left, right = write_run(tmp_path, "left"), write_run(tmp_path, "right")
    path = right / "study_run.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        paired_comparison(left, right, iterations=2)


def test_missing_truth_and_legacy_provenance_are_rejected(tmp_path):
    left, right = write_run(tmp_path, "left"), write_run(tmp_path, "right")
    table = pd.read_csv(right / "predictions_dev.csv")
    table["y_true"] = None
    table.to_csv(right / "predictions_dev.csv", index=False)
    with pytest.raises(ValueError, match="targets"):
        paired_comparison(left, right, iterations=2)
    (right / "study_run.json").unlink()
    with pytest.raises(ValueError, match="Legacy"):
        paired_comparison(left, right, iterations=2)
