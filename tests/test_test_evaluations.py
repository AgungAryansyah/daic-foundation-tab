from __future__ import annotations

import json

import pandas as pd
import yaml

from daic_foundation_tab.tracking.test_evaluations import collect_test_evaluations


def test_collect_test_evaluations_backfills_runs_without_duplicates(tmp_path) -> None:
    for name, feature_set, score in (
        ("run_audio", "audio", 0.5),
        ("run_visual", "visual", 0.75),
    ):
        run = tmp_path / name
        run.mkdir()
        config = {
            "project": {"seed": 42},
            "experiment": {"name": f"edaic_{feature_set}", "feature_set": feature_set},
            "evaluation": {"threshold": 0.25 if feature_set == "visual" else 0.5},
            "data": {"root": "data/edaic"},
            "model": {"name": "tabiclv2_ft"},
        }
        (run / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (run / "metrics_test.json").write_text(
            json.dumps(
                {"macro_f1": score, "accuracy": score, "confusion_matrix": [[1, 0], [0, 1]]}
            ),
            encoding="utf-8",
        )
        (run / "metrics_test.csv").write_text(
            f"macro_f1,accuracy\n{score},{score}\n", encoding="utf-8"
        )
        if feature_set == "visual":
            (run / "decision_threshold.json").write_text('{"threshold": 0.25}', encoding="utf-8")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "metrics_test.json").write_text('{"macro_f1": 0.9}', encoding="utf-8")

    destination = collect_test_evaluations(tmp_path)
    first = destination.read_text(encoding="utf-8")
    collect_test_evaluations(tmp_path)
    comparison = pd.read_csv(destination)

    assert destination.read_text(encoding="utf-8") == first
    assert comparison["run"].tolist() == ["run_audio", "run_visual"]
    assert comparison["experiment"].tolist() == ["edaic_audio", "edaic_visual"]
    assert comparison["dataset"].tolist() == ["edaic", "edaic"]
    assert comparison["feature_set"].tolist() == ["audio", "visual"]
    assert comparison["threshold"].tolist() == [0.5, 0.25]
    assert comparison["macro_f1"].tolist() == [0.5, 0.75]
    assert "confusion_matrix" not in comparison.columns
