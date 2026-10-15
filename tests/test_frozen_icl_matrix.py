from __future__ import annotations

from collections import Counter

import pytest

from daic_foundation_tab import matrix

from .test_tabpfn35_matrix import _fake_execution


def test_all_family_frozen_icl_matrix_runs_all_smokes_before_full_runs(monkeypatch, tmp_path):
    calls, status = _fake_execution(monkeypatch, tmp_path)

    matrix.run_matrix(status_path=status, model_family="all", mode="frozen_icl")

    assert len(calls) == 24
    assert all(path.stem.endswith("_smoke") for path in calls[:6])
    assert all(not path.stem.endswith("_smoke") for path in calls[6:])
    assert Counter(path.name.split("_icl_", 1)[0] for path in calls) == {
        "tabiclv2": 8,
        "tabpfn35": 8,
        "kumo_medium": 8,
    }
    assert all("_icl_" in path.name for path in calls)


def test_all_family_matrix_is_frozen_icl_only():
    with pytest.raises(ValueError, match="available for frozen_icl"):
        matrix.run_matrix(model_family="all", mode="fine_tuning")
