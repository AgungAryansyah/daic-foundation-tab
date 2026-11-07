import json
import shutil

import pandas as pd
import pytest

from daic_foundation_tab.transcript_study import groups
from daic_foundation_tab.transcript_study.bundle import read_bundle
from daic_foundation_tab.transcript_study.collection import collect_study
from daic_foundation_tab.transcript_study.comparisons import compare_cv
from tests.test_transcript_study_audit import prepared_input
from tests.test_transcript_study_groups import host_input, install_model


def completed_campaign(tmp_path, monkeypatch, *, stage="cv", family="qwen35_2b"):
    bundle, host = prepared_input(tmp_path, family), host_input(tmp_path)
    install_model(monkeypatch)
    monkeypatch.setattr(groups, "_run_subprocess", lambda bundle, path, condition, host:
                        groups.condition_worker(bundle, path, condition["condition_id"], condition["stage"]))
    for task in ("classification", "regression"):
        groups.run_group(bundle, host, task, stage=stage)
    return bundle, tmp_path / "runs"


@pytest.mark.parametrize("family", ["qwen35_2b", "qwen35_0_8b"])
def test_complete_collection_covers_each_participant_once_and_pairs_methods(tmp_path, monkeypatch, family):
    bundle, runs = completed_campaign(tmp_path, monkeypatch, family=family)
    report = collect_study(bundle, [runs], tmp_path / "report")
    coverage = json.loads((report / "coverage.json").read_text())
    assert coverage["status"] == "completed" and coverage["completed"] == 20
    metrics = pd.read_csv(report / "condition_metrics.csv")
    assert len(metrics) == 20 and set(metrics["family"]) == {family}
    assert (runs / read_bundle(bundle)[0]["study_id"] / "cv" / f"{family}-classification").is_dir()
    for record in runs.rglob("study_run.json"):
        assert json.loads(record.read_text())["execution_group"].startswith(family)
    for path in (report / "oof_predictions").glob("*.csv"):
        table = pd.read_csv(path)
        assert len(table) == 30 and not table["participant_id"].duplicated().any()
        assert set(table["fold"]) == {1, 2, 3, 4, 5}
    differences = pd.read_csv(report / "cv_paired_oof.csv")
    assert differences["delta_ft_minus_icl"].eq(0).all()
    assert differences["iterations"].eq(5).all()
    assert "cpu_threads" in pd.read_csv(report / "condition_metrics.csv")
    duplicate = tmp_path / "duplicate"
    shutil.copytree(runs, duplicate)
    with pytest.raises(ValueError, match="Duplicate successful"):
        collect_study(bundle, [runs, duplicate], tmp_path / "duplicate-report")


def test_collector_rejects_missing_conditions_and_changed_results(tmp_path, monkeypatch):
    bundle = prepared_input(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        collect_study(bundle, [runs], tmp_path / "report")
    report = collect_study(bundle, [runs], tmp_path / "partial", allow_partial=True)
    assert json.loads((report / "coverage.json").read_text())["status"] == "incomplete"


def test_transcript_comparison_validates_report_and_imported_folds(tmp_path, monkeypatch):
    bundle, runs = completed_campaign(tmp_path, monkeypatch)
    report = collect_study(bundle, [runs], tmp_path / "report")
    reference = (bundle, report, "frozen_icl")
    candidate = (bundle, report, "fine_tuning")
    result = compare_cv(reference, candidate, tmp_path / "paired")
    table = pd.read_csv(result)
    assert table["metric"].tolist() == ["roc_auc", "mae"]
    assert table["candidate_adaptation"].eq("lora").all()
    assert table["delta_candidate_minus_reference"].eq(0).all()
    summary = pd.read_csv(report / "cv_summary.csv")
    path = report / summary.iloc[0]["predictions_file"]
    predictions = pd.read_csv(path)
    predictions.loc[0, "fold"] = 99
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="imported participants, targets, and folds"):
        compare_cv(reference, candidate, tmp_path / "wrong-folds")


def test_final_collection_retains_unlabeled_test_predictions(tmp_path, monkeypatch):
    bundle, runs = completed_campaign(tmp_path, monkeypatch, stage="final")
    report = collect_study(bundle, [runs], tmp_path / "report", stage="final")
    assert json.loads((report / "coverage.json").read_text())["completed"] == 4
    differences = pd.read_csv(report / "final_differences.csv")
    assert set(differences["split"]) == {"dev"}


@pytest.mark.parametrize("family", ["qwen35_2b", "qwen35_0_8b"])
def test_cross_input_comparison_requires_exact_parent_participants_targets_and_folds(tmp_path, monkeypatch, family):
    bundle, runs = completed_campaign(tmp_path, monkeypatch, family=family)
    transcript_report = collect_study(bundle, [runs], tmp_path / "report")
    tabular = tmp_path / "tabular-report"
    shutil.copytree(transcript_report, tabular)
    coverage = json.loads((tabular / "coverage.json").read_text())
    coverage["study_id"] = read_bundle(bundle)[0]["parent_study_id"]
    (tabular / "coverage.json").write_text(json.dumps(coverage))
    summary = pd.read_csv(tabular / "cv_summary.csv")
    summary["family"], summary["modality"] = "tabpfn35", "audio"
    summary.to_csv(tabular / "cv_summary.csv", index=False)
    report = collect_study(bundle, [runs], tmp_path / "matched", tabular_report=tabular)
    comparisons = pd.read_csv(report / "cross_input_comparisons.csv")
    assert set(comparisons["transcript_family"]) == {family}
    assert comparisons["delta_transcript_minus_tabular"].eq(0).all()
    assert comparisons["interpretation"].eq("model and input both differ").all()
    path = tabular / summary.iloc[0]["predictions_file"]
    predictions = pd.read_csv(path)
    predictions.loc[0, "fold"] = 99
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="different folds"):
        collect_study(bundle, [runs], tmp_path / "wrong-folds", tabular_report=tabular)
    assert json.loads((tmp_path / "wrong-folds/coverage.json").read_text())["status"] == "failed"
