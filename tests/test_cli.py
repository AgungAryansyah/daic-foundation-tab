from __future__ import annotations

import json
from pathlib import Path

from daic_foundation_tab.cli import _compare


def test_compare_writes_csv_and_markdown_without_optional_dependencies(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics_dev.json").write_text(
        json.dumps({"macro_f1": 0.5, "confusion_matrix": [[1, 0], [0, 1]]}), encoding="utf-8"
    )

    output = _compare([run], tmp_path / "comparison")

    assert (output / "comparison.csv").is_file()
    assert (output / "comparison.md").read_text(encoding="utf-8") == "| run | macro_f1 |\n| --- | --- |\n| run | 0.5 |\n"
