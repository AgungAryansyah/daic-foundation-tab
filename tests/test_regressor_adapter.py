from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.models.registry import create_model
from daic_foundation_tab.models.tabiclv2_ft import TabICLv2FineTuningError
from daic_foundation_tab.models.tabiclv2_ft_regressor import TabICLv2FineTunedRegressor


class _FakeRegressor:
    instances: ClassVar[list[_FakeRegressor]] = []

    def __init__(self, **parameters) -> None:
        self.parameters = parameters
        self._best_metric_ = -2.25
        self.fit_arguments = None
        self.instances.append(self)

    def fit(self, features, target, X_val, y_val, output_dir) -> None:
        self.fit_arguments = (features.copy(), target.copy(), X_val.copy(), y_val.copy())
        (output_dir / "best.ckpt").write_bytes(b"regression checkpoint")

    def predict(self, features):
        return np.full(len(features), 12.5)


def test_regressor_uses_validation_data_and_reports_positive_mae(monkeypatch, tmp_path) -> None:
    _FakeRegressor.instances.clear()
    monkeypatch.setitem(
        sys.modules, "tabicl", SimpleNamespace(FinetunedTabICLRegressor=_FakeRegressor)
    )
    model = create_model(
        {
            "name": "tabiclv2_ft_regressor",
            "checkpoint_version": "regressor.ckpt",
            "parameters": {"eval_metric": "mae", "amp": True, "device": "cuda:0"},
        }
    )
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([1.0, 5.0, 12.0, 18.0])
    reported = []

    model.fit(
        features.iloc[:2],
        target.iloc[:2],
        validation_features=features.iloc[2:],
        validation_target=target.iloc[2:],
        checkpoint_directory=tmp_path / "checkpoints",
        epoch_callback=lambda metrics: reported.append(dict(metrics)),
    )

    assert isinstance(model, TabICLv2FineTunedRegressor)
    instance = _FakeRegressor.instances[0]
    assert instance.parameters["checkpoint_version"] == "regressor.ckpt"
    assert instance.fit_arguments[3].tolist() == [12.0, 18.0]
    instance._make_experiment_logger().log_epoch({"train/epoch": 0, "val/mae": 2.25}, step=1)
    assert reported == [{"train/epoch": 0, "val/mae": 2.25}]
    assert model.predict(features).tolist() == [12.5] * 4
    assert model.finetune_metadata()["best_validation_metric"] == 2.25
    assert model.run_metadata()["model_name"] == "TabICLv2-FT Regressor"


def test_regressor_rejects_non_numeric_targets_before_loading_tabicl(tmp_path) -> None:
    model = TabICLv2FineTunedRegressor(
        {
            "checkpoint_version": "regressor.ckpt",
            "parameters": {"eval_metric": "mae", "amp": True, "device": "cuda:0"},
        }
    )
    features = pd.DataFrame({"feature": [0.0, 1.0]})

    with pytest.raises(TabICLv2FineTuningError, match="finite"):
        model.fit(
            features,
            pd.Series([1.0, float("nan")]),
            validation_features=features,
            validation_target=pd.Series([2.0, 3.0]),
            checkpoint_directory=tmp_path / "checkpoints",
        )
