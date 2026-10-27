import shutil

import pytest

from daic_foundation_tab.tabular_study.bundle import read_bundle
from daic_foundation_tab.tabular_study.condition import run_condition
from daic_foundation_tab.tabular_study.results import verify_result, write_result

pytest_plugins = ["tests.test_tabular_study_condition"]


def test_verified_result_is_portable_without_checkpoint(prepared, tmp_path):
    bundle, _ = prepared
    context = read_bundle(bundle)
    condition = next(
        x for x in context[0]["conditions"]["cv"] if x["mode"] == "fine_tuning" and x["fold"] == 1
    )
    output = run_condition(bundle, condition, {"output_root": str(tmp_path / "runs")}, "first")
    result = write_result(context, condition, output)
    assert result["checkpoint"]["sha256"]
    relocated = tmp_path / "copy"
    shutil.copytree(output, relocated, ignore=shutil.ignore_patterns("checkpoints"))
    assert verify_result(context, relocated)["condition"] == condition
    with pytest.raises(ValueError, match="checkpoint"):
        verify_result(context, relocated, require_checkpoint=True)
    (relocated / "predictions_cv.csv").write_text("tampered")
    with pytest.raises(ValueError, match="artifact"):
        verify_result(context, relocated)
