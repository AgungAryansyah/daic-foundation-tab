from __future__ import annotations

import hashlib
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from daic_foundation_tab.models.registry import create_model
from daic_foundation_tab.models.tabpfn35_ft import TabPFN35FineTuningError


class _Version(str, Enum):
    V3_5 = "v3.5"


def _install_fine_tuner(monkeypatch, metrics):
    instances = []
    exports = []

    class FineTuner:
        def __init__(self, **parameters):
            self.parameters = parameters
            self.min_delta = parameters.get("min_delta", 0.0001)
            self.fit_arguments = None
            instances.append(self)

        def _is_improvement(self, current, best):
            if self.parameters["eval_metric"] == "mse":
                return current < best - self.min_delta
            return current > best + self.min_delta

        def _get_checkpoint_metrics(self, result):
            return {self.parameters["eval_metric"]: result.primary}

        def _log_epoch_evaluation(self, epoch, result, mean_train_loss):
            pass

        def fit(self, features, target, *, X_val, y_val, output_dir):
            self.fit_arguments = (features, target, X_val, y_val, output_dir)
            best = metrics[0]
            selected_epoch = 0
            counter = 0
            for epoch, metric in enumerate(metrics, start=-1):
                self._log_epoch_evaluation(epoch, SimpleNamespace(primary=metric), 0.8)
                if epoch == -1:
                    continue
                logger = self.parameters.get("experiment_logger")
                if logger is not None:
                    logger.log_step({"train/lr": 1e-5}, epoch + 1)
                    metric_name = "MSE" if self.parameters["eval_metric"] == "mse" else "ROC AUC"
                    logger.log_epoch(
                        {"train/epoch": epoch, "train/mean_loss": 0.8, f"val/{metric_name}": metric},
                        epoch + 1,
                    )
                if self._is_improvement(metric, best):
                    best = metric
                    selected_epoch = epoch + 1
                    counter = 0
                else:
                    counter += 1
                if counter >= self.parameters.get("early_stopping_patience", 10):
                    break
            self.finetuned_estimator_ = SimpleNamespace(selected_epoch=selected_epoch)

        def predict(self, features):
            return np.full(len(features), 12.5 if self.parameters["eval_metric"] == "mse" else 0)

        def predict_proba(self, features):
            return np.tile([0.7, 0.3], (len(features), 1))

    def export(estimator, destination, *, additional_fields):
        exports.append((estimator, destination, additional_fields))
        destination.write_bytes(b"selected weights")

    monkeypatch.setitem(sys.modules, "tabpfn", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tabpfn.constants", SimpleNamespace(ModelVersion=_Version))
    monkeypatch.setitem(
        sys.modules,
        "tabpfn.finetuning",
        SimpleNamespace(FinetunedTabPFNClassifier=FineTuner, FinetunedTabPFNRegressor=FineTuner),
    )
    monkeypatch.setitem(
        sys.modules,
        "tabpfn.finetuning.train_util",
        SimpleNamespace(get_checkpoint_name=lambda n, is_best: f"checkpoint_{n}_best.pth"),
    )
    monkeypatch.setitem(sys.modules, "tabpfn.model_loading", SimpleNamespace(save_tabpfn_model=export))
    return instances, exports


def _model(regression=False, **parameters):
    return create_model(
        {
            "name": "tabpfn35_ft_regressor" if regression else "tabpfn35_ft",
            "model_version": "v3.5",
            "checkpoint_version": "tabpfn-v3.5-20260909.safetensors",
            "parameters": {
                "device": "cuda:0",
                "eval_metric": "mse" if regression else "roc_auc",
                "early_stopping": True,
                **parameters,
            },
        }
    )


def _fit(model, directory, callback=None, target=None):
    features = pd.DataFrame({"feature": np.arange(8, dtype=float)})
    target = pd.Series([0, 1] * 4) if target is None else target
    return model.fit(
        features.iloc[:6],
        target.iloc[:6],
        validation_features=features.iloc[6:],
        validation_target=target.iloc[6:],
        checkpoint_directory=directory,
        epoch_callback=callback,
    )


@pytest.mark.parametrize(
    "regression,metrics,selected_epoch,selected_metric",
    [
        (False, [0.75, 0.7, 0.74], 0, 0.75),
        (False, [0.75, 0.8, 0.76], 1, 0.8),
        (False, [0.75, 0.75005, 0.7501], 0, 0.75),
        (True, [3.0, 3.5, 4.0], 0, 3.0),
        (True, [3.0, 2.0, 2.5], 1, 2.0),
    ],
)
def test_native_selection_exports_restored_weights_and_tracks_epochs(
    monkeypatch, tmp_path, regression, metrics, selected_epoch, selected_metric
):
    instances, exports = _install_fine_tuner(monkeypatch, metrics)
    model = _model(regression)
    reported = []
    _fit(model, tmp_path / "checkpoints", reported.append)

    instance = instances[0]
    assert instance.parameters["model_version"] is _Version.V3_5
    assert "amp" not in instance.parameters
    assert len(instance.fit_arguments[0]) == 6
    assert len(instance.fit_arguments[2]) == 2
    assert instance.fit_arguments[3].tolist() == [0, 1]
    metric = "mse" if regression else "roc_auc"
    assert reported[0] == {
        "train/epoch": 0,
        "train/mean_loss": 0.8,
        f"val/{metric}": metrics[1],
        "train/lr": 1e-5,
    }
    assert exports[0][0].selected_epoch == selected_epoch
    assert exports[0][2]["epoch"] == selected_epoch
    metadata = model.finetune_metadata()
    assert metadata["selection_metric"] == metric
    assert metadata["baseline_validation_metric"] == metrics[0]
    assert metadata["best_validation_metric"] == selected_metric
    assert metadata["selected_epoch"] == selected_epoch
    assert Path(metadata["checkpoint_path"]).name == "checkpoint_6_best.pth"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(b"selected weights").hexdigest()
    assert model.predict(instance.fit_arguments[0]).shape == (6,)
    if not regression:
        assert model.predict_proba(instance.fit_arguments[0]).shape == (6, 2)
    assert model.run_metadata()["model_version"] == "v3.5"


def test_early_stopping_keeps_selected_epoch_and_only_reports_completed_epochs(monkeypatch, tmp_path):
    _install_fine_tuner(monkeypatch, [0.75, 0.8, 0.7, 0.6, 0.9])
    model = _model(early_stopping_patience=2)
    reported = []
    _fit(model, tmp_path, reported.append)
    assert len(reported) == 3
    assert model.finetune_metadata()["selected_epoch"] == 1


def test_nonfinite_baseline_fails_without_exporting_checkpoint(monkeypatch, tmp_path):
    _, exports = _install_fine_tuner(monkeypatch, [float("nan"), 0.7])
    model = _model()
    with pytest.raises(TabPFN35FineTuningError, match="Initial validation metric"):
        _fit(model, tmp_path)
    assert not exports
    with pytest.raises(TabPFN35FineTuningError, match="Call fit"):
        model.predict(pd.DataFrame({"feature": [1]}))


def test_nonfinite_epoch_keeps_finite_baseline(monkeypatch, tmp_path):
    _install_fine_tuner(monkeypatch, [0.75, float("nan")])
    model = _model()
    _fit(model, tmp_path)
    assert model.finetune_metadata()["best_validation_metric"] == 0.75


def test_invalid_targets_and_existing_checkpoints_are_rejected_before_training(tmp_path):
    with pytest.raises(TabPFN35FineTuningError, match="finite"):
        _fit(_model(True), tmp_path, target=pd.Series([np.nan] * 8))
    (tmp_path / "checkpoint_6_best.pth").write_bytes(b"existing")
    with pytest.raises(TabPFN35FineTuningError, match="fresh checkpoint directory"):
        _fit(_model(), tmp_path)
