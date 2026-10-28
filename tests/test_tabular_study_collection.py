import json
import shutil

import pandas as pd
import pytest

from daic_foundation_tab.tabular_study import condition as execution
from daic_foundation_tab.tabular_study.bundle import read_bundle
from daic_foundation_tab.tabular_study.collection import collect_study
from daic_foundation_tab.tabular_study.condition import run_condition
from daic_foundation_tab.tabular_study.results import write_result

pytest_plugins = ["tests.test_tabular_study_condition"]


def test_collects_distributed_results_and_rejects_incomplete_or_conflicting_attempts(
    prepared, tmp_path
):
    bundle, _ = prepared
    context = read_bundle(bundle)
    roots = [tmp_path / "machine-a", tmp_path / "machine-b"]
    first_output = None
    for index, item in enumerate(context[0]["conditions"]["cv"]):
        root = roots[index % 2]
        output = run_condition(
            bundle,
            item,
            {"output_root": str(root), "host_label": f"host-{index % 2}"},
            f"attempt-{index}",
        )
        write_result(context, item, output)
        first_output = first_output or output
    report = collect_study(bundle, roots, tmp_path / "report")
    coverage = json.loads((report / "coverage.json").read_text())
    assert coverage["status"] == "completed" and coverage["completed"] == 120
    assert len(pd.read_csv(report / "condition_metrics.csv")) == 120
    assert len(list((report / "oof_predictions").glob("*.csv"))) == 24
    pairs = pd.read_csv(report / "cv_paired_oof.csv")
    fold_pairs = pd.read_csv(report / "cv_fold_differences.csv")
    assert len(pairs.loc[pairs["metric"] == "macro_f1"]) == 6
    assert len(pairs.loc[pairs["metric"] == "mae"]) == 6
    assert len(fold_pairs.loc[fold_pairs["metric"] == "macro_f1"]) == 30
    assert len(fold_pairs.loc[fold_pairs["metric"] == "mae"]) == 30
    assert pairs["delta_ft_minus_icl"].eq(0).all()
    summary = pd.read_csv(report / "cv_summary.csv")
    assert summary["participants"].eq(30).all()
    assert summary["valid_folds"].max() == 5
    copy = tmp_path / "copied"
    shutil.copytree(first_output, copy, ignore=shutil.ignore_patterns("checkpoints"))
    partial = collect_study(bundle, [copy], tmp_path / "partial", allow_partial=True)
    assert json.loads((partial / "coverage.json").read_text())["status"] == "incomplete"
    assert not list((partial / "oof_predictions").glob("*.csv"))
    with pytest.raises(ValueError, match="incomplete"):
        collect_study(bundle, [copy], tmp_path / "strict")
    duplicate = tmp_path / "duplicate"
    item = context[0]["conditions"]["cv"][0]
    another = run_condition(bundle, item, {"output_root": str(duplicate)}, "another-attempt")
    write_result(context, item, another)
    with pytest.raises(ValueError, match="Conflicting"):
        collect_study(bundle, [first_output, duplicate], tmp_path / "conflict", allow_partial=True)
    assert "conflicting" in {
        row["status"]
        for row in json.loads((tmp_path / "conflict/coverage.json").read_text())["conditions"]
    }


def test_collection_rejects_inconsistent_pretrained_checkpoints(prepared, tmp_path, monkeypatch):
    bundle, _ = prepared
    context = read_bundle(bundle)
    monkeypatch.setattr(execution, "_source_checkpoint_hash",
                        lambda condition, detail: ("a" if condition["mode"] == "frozen_icl" else "b") * 64)
    root = tmp_path / "runs"
    items = [x for x in context[0]["conditions"]["cv"] if x["family"] == "tabpfn35"
             and x["modality"] == "audio" and x["task"] == "classification" and x["fold"] == 1]
    for index, item in enumerate(items):
        output = run_condition(bundle, item, {"output_root": str(root)}, f"attempt-{index}")
        write_result(context, item, output)
    with pytest.raises(ValueError, match="checkpoints differ"):
        collect_study(bundle, [root], tmp_path / "report", allow_partial=True)


def test_optional_final_stage_compares_development_and_labeled_test(prepared, tmp_path):
    bundle, _ = prepared
    context = read_bundle(bundle)
    root = tmp_path / "runs"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"Participant_ID": [500, 501], "PHQ8_Score": [3, 14]}).to_csv(truth, index=False)
    items = [x for x in context[0]["conditions"]["final"] if x["family"] == "tabpfn35"
             and x["modality"] == "audio" and x["task"] == "classification"]
    for index, item in enumerate(items):
        output = run_condition(bundle, item, {"output_root": str(root), "test_ground_truth": str(truth)}, f"attempt-{index}")
        write_result(context, item, output)
    report = collect_study(bundle, [root], tmp_path / "final-report", stage="final", allow_partial=True)
    differences = pd.read_csv(report / "final_differences.csv")
    assert set(differences["split"]) == {"dev", "test"}
    assert len(differences.loc[differences["metric"] == "macro_f1"]) == 2
