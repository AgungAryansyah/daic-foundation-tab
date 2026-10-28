import json
from copy import deepcopy

import pandas as pd
import pytest

from daic_foundation_tab.tabular_study import bundle as bundles
from daic_foundation_tab.tabular_study import condition as execution
from daic_foundation_tab.tabular_study import revisions
from daic_foundation_tab.tabular_study.bundle import read_bundle, verify_bundle
from daic_foundation_tab.tabular_study.collection import collect_study
from daic_foundation_tab.tabular_study.results import verify_result, write_result

pytest_plugins = ["tests.test_tabular_study_condition"]
MODEL_PATH = "src/daic_foundation_tab/models/kumo_medium_ft.py"


def simulated_change(monkeypatch, path=MODEL_PATH):
    before = revisions._working_files()
    after = deepcopy(before)
    after[path] = "b" * 64
    monkeypatch.setattr(revisions, "_working_files", lambda: after)
    monkeypatch.setattr(revisions, "_git_files", lambda commit: before if commit == "baseline" else after)
    identity = revisions._identity(after)
    monkeypatch.setattr(bundles, "software_identity", lambda: identity)
    monkeypatch.setattr(execution, "software_identity", lambda: identity)
    return identity


def test_revision_preserves_old_results_and_records_new_source(prepared, tmp_path, monkeypatch):
    directory, _ = prepared
    context = read_bundle(directory)
    original_definition = (directory / "study.json").read_bytes()
    old_condition, new_condition = context[0]["conditions"]["cv"][:2]
    old_output = execution.run_condition(directory, old_condition, {"output_root": str(tmp_path / "runs")}, "old")
    original = write_result(context, old_condition, old_output)
    identity = simulated_change(monkeypatch)
    with pytest.raises(ValueError, match="Executed source"):
        read_bundle(directory)
    with pytest.raises(ValueError, match="declared source changes"):
        revisions.register_revision(directory, "baseline", ["src/unrelated.py"], "AMP repair")
    assert not (directory / revisions.FILENAME).exists()
    record = revisions.register_revision(directory, "baseline", [MODEL_PATH], "AMP overflow recovery")
    updated = read_bundle(directory)
    assert verify_result(updated, old_output) == original
    new_output = execution.run_condition(directory, new_condition, {"output_root": str(tmp_path / "runs")}, "new")
    result = write_result(updated, new_condition, new_output)
    assert result["software_identity"] == identity != original["software_identity"]
    lifecycle = json.loads((new_output / "study_run.json").read_text())
    assert lifecycle["software_revision_id"] == record["revision_id"]
    report = collect_study(directory, [tmp_path / "runs"], tmp_path / "report", allow_partial=True)
    metrics = pd.read_csv(report / "condition_metrics.csv")
    assert set(metrics["software_identity"]) == {identity, original["software_identity"]}
    assert json.loads((report / "software_provenance.json").read_text())["execution_revisions"][identity]["reason"] == "AMP overflow recovery"
    assert (directory / "study.json").read_bytes() == original_definition
    assert revisions.register_revision(directory, "baseline", [MODEL_PATH], "same revision") == record
    manifest = json.loads((directory / revisions.FILENAME).read_text())
    manifest["revisions"][0]["reason"] = "tampered"
    (directory / revisions.FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision is invalid"):
        verify_bundle(directory)


def test_revision_rejects_dependency_changes(prepared, monkeypatch):
    directory, _ = prepared
    simulated_change(monkeypatch, "uv.lock")
    with pytest.raises(ValueError, match="declared source changes"):
        revisions.register_revision(directory, "baseline", ["uv.lock"], "dependency change")
    assert not (directory / revisions.FILENAME).exists()


def test_git_snapshot_fingerprint_matches_original_implementation():
    assert revisions._identity(revisions._working_files()) == bundles.software_identity()
