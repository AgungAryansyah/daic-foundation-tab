from __future__ import annotations

import json
import sys
from pathlib import Path

from daic_foundation_tab.cli import _compare, _validate, main

from .helpers import write_synthetic_dataset


def test_compare_writes_csv_and_markdown_without_optional_dependencies(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics_dev.json").write_text(
        json.dumps({"macro_f1": 0.5, "confusion_matrix": [[1, 0], [0, 1]]}), encoding="utf-8"
    )

    output = _compare([run], tmp_path / "comparison")

    assert (output / "comparison.csv").is_file()
    assert (output / "comparison.md").read_text(encoding="utf-8") == "| run | macro_f1 |\n| --- | --- |\n| run | 0.5 |\n"


def test_validate_regression_reports_phq8_score_distribution(tmp_path: Path) -> None:
    config = write_synthetic_dataset(tmp_path / "data")
    config["experiment"]["task"] = "regression"

    output = _validate(config)

    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert report["split_statistics"]["train"]["phq8_score"]["min"] == 2.0
    assert "phq8_score" not in report["split_statistics"]["test"]


def test_collect_regression_test_results_command(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["daic-foundation-tab", "collect-regression-test-results", "--output-root", str(tmp_path)],
    )

    main()

    assert capsys.readouterr().out.strip() == str(tmp_path / "regression_test_evaluations.csv")
    assert (tmp_path / "regression_test_evaluations.csv").is_file()
