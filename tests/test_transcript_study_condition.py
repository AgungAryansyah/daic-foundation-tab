import json

import pandas as pd
import pytest

from daic_foundation_tab import text_runner
from daic_foundation_tab.transcript_study.bundle import read_bundle, roles_for
from daic_foundation_tab.transcript_study.condition import run_condition
from tests.test_text_runner import FakeTextModel, mock_gpu
from tests.test_transcript_study_audit import prepared_input


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["frozen_icl", "fine_tuning"])
def test_fold_runner_uses_exact_assignments_without_evaluation_labels(tmp_path, monkeypatch, task, mode):
    bundle = prepared_input(tmp_path)
    study, rows, _, folds = read_bundle(bundle)
    condition = next(c for c in study["conditions"]["cv"] if c["task"] == task and c["mode"] == mode)
    roles = roles_for(rows, folds, condition)
    mock_gpu(monkeypatch)
    model = FakeTextModel(task)
    original_fit = model.fit

    def fit(training, validation, *args, **kwargs):
        assert [x.participant_id for x in training] == roles["fit"]
        assert [x.participant_id for x in validation] == roles["selection"]
        return original_fit(training, validation, *args, **kwargs)

    model.fit = fit
    monkeypatch.setattr(text_runner, "create_text_model", lambda config: model)
    host = {"output_root": str(tmp_path / "runs"), "host_label": "test", "wandb": {"mode": "disabled"}}
    output = run_condition(bundle, condition, host)
    predictions = pd.read_csv(output / "predictions_cv.csv", dtype={"participant_id": str})
    assert predictions["participant_id"].tolist() == roles["evaluation"]
    assert set(predictions["split"]) == {"cv"}
    assert not (output / "predictions_dev.csv").exists()
    assert {x.participant_id for x in model.demonstrations} <= set(roles["fit"])
    record = json.loads((output / "study_run.json").read_text())
    assert record["condition_id"] == condition["condition_id"]
    assert record["selection_labels_used"] == (mode == "fine_tuning")
    assert record["status"] == "completed"
    assert (output / "metrics_cv.json").is_file()


def test_final_stage_uses_official_dev_and_unlabeled_test(tmp_path, monkeypatch):
    bundle = prepared_input(tmp_path)
    study = read_bundle(bundle)[0]
    condition = study["conditions"]["final"][0]
    mock_gpu(monkeypatch)
    monkeypatch.setattr(text_runner, "create_text_model", lambda config: FakeTextModel(condition["task"]))
    output = run_condition(bundle, condition, {"output_root": str(tmp_path / "runs"), "host_label": "test"})
    predictions = pd.read_csv(output / "predictions_test.csv")
    assert predictions["y_true"].isna().all()
    assert not (output / "metrics_test.json").exists()
    assert (output / "metrics_dev.json").is_file()
