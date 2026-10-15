from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from daic_foundation_tab.config import load_config
from daic_foundation_tab.data.dataset import ParticipantDataset
from daic_foundation_tab.evaluation.fine_tuning import stratified_fine_tune_partition
from daic_foundation_tab.evaluation.icl_holdout import repeated_icl_holdout
from daic_foundation_tab.evaluation.icl_splits import prepare_icl_splits
from daic_foundation_tab.evaluation.regression_splits import regression_fine_tune_partition
from daic_foundation_tab.models import create_model
from daic_foundation_tab.models.icl import FrozenICLClassifier, FrozenICLModel
from daic_foundation_tab.models.kumo_medium_ft import PACKAGE_REVISION

_ROOT = Path(__file__).parents[1]
_ICL_PRESETS = sorted((_ROOT / "configs/experiments").glob("*_icl_*.yaml"))


@pytest.mark.parametrize("preset", _ICL_PRESETS, ids=lambda path: path.stem)
def test_frozen_icl_presets_are_inference_only_and_cover_the_full_matrix(preset):
    config = load_config(preset)
    experiment = config["experiment"]
    parameters = config["model"]["parameters"]

    assert experiment["mode"] == "frozen_icl"
    assert config["data"]["root"] == "data/edaic"
    assert config["runtime"] == {"device": "cuda:0", "require_cuda": True}
    assert parameters == {"n_estimators": 8, "device": "cuda:0", "random_state": 42}
    assert config["evaluation"]["test_predictions"] is True
    assert config["evaluation"]["fine_tuning"]["validation_fraction"] == 0.2
    smoke = preset.stem.endswith("_smoke")
    assert experiment.get("profile") == ("smoke" if smoke else None)
    assert config["bootstrap"]["enabled"] is not smoke
    assert config["evaluation"]["repeated_holdout"]["enabled"] is not smoke
    if not smoke:
        assert config["bootstrap"]["iterations"] == 2_000
        assert config["evaluation"]["repeated_holdout"]["repeats"] == 30
    assert config["modalities"]["audio"]["enabled"] == (
        experiment["feature_set"] in {"audio", "audio_visual"}
    )
    assert config["modalities"]["visual"]["enabled"] == (
        experiment["feature_set"] in {"visual", "audio_visual"}
    )


def _split_dataset(task: str, random_state: int = 42) -> tuple[ParticipantDataset, pd.Series]:
    if task == "classification":
        target = pd.Series(np.tile([0, 1], 10))
        partition = stratified_fine_tune_partition(
            target,
            pd.Series([f"train_{index}" for index in range(len(target))]),
            0.2,
            random_state,
            training_role="icl_context",
            validation_role="reserved_validation",
        )
    else:
        target = pd.Series(np.arange(20, dtype=float))
        partition = regression_fine_tune_partition(
            target,
            pd.Series([f"train_{index}" for index in range(len(target))]),
            0.2,
            random_state,
            training_role="icl_context",
            validation_role="reserved_validation",
        )
    holdout_constant = np.zeros(len(target), dtype=float)
    holdout_constant[partition.validation_indices] = 1
    ids = [*(f"train_{index}" for index in range(len(target))), "dev_0", "dev_1", "test_0"]
    table = pd.DataFrame(
        {
            "participant_id": ids,
            "split": ["train"] * len(target) + ["dev", "dev", "test"],
            "target_binary": [*(target.astype(int) if task == "classification" else target >= 10), 0, 1, pd.NA],
            "target_phq8": [*(target.astype(float) if task == "regression" else target * 5.0), 2.0, 14.0, np.nan],
            "signal": [*np.arange(len(target), dtype=float), 99.0, 100.0, 101.0],
            "holdout_constant": [*holdout_constant, 99.0, 100.0, 101.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "column": ["signal", "holdout_constant"],
            "modality": ["audio", "audio"],
            "source_group": ["test", "test"],
            "source_feature": ["signal", "holdout_constant"],
            "aggregation": ["mean", "mean"],
        }
    )
    return ParticipantDataset(table, manifest, pd.DataFrame(), {}, {}, "test"), target


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_icl_context_split_matches_the_fine_tuning_partition_and_filters_on_context(task):
    dataset, target = _split_dataset(task)
    prepared = prepare_icl_splits(dataset, "audio", task, 0.2, 42)
    expected = (
        stratified_fine_tune_partition(
            target,
            pd.Series([f"train_{index}" for index in range(len(target))]),
            0.2,
            42,
            training_role="icl_context",
            validation_role="reserved_validation",
        )
        if task == "classification"
        else regression_fine_tune_partition(
            target,
            pd.Series([f"train_{index}" for index in range(len(target))]),
            0.2,
            42,
            training_role="icl_context",
            validation_role="reserved_validation",
        )
    )

    pd.testing.assert_frame_equal(prepared.assignment, expected.assignment, check_dtype=False)
    assert len(prepared.context_x) == len(expected.training_indices)
    assert len(prepared.reserved_x) == len(expected.validation_indices)
    assert prepared.selection.columns == ["signal"]
    assert set(prepared.assignment["role"]) == {"icl_context", "reserved_validation"}


def _model_config(name: str, checkpoint: Path) -> dict:
    return {
        "name": name,
        "checkpoint_version": checkpoint.name,
        "parameters": {"n_estimators": 8, "device": "cuda:0", "random_state": 42},
    }


@pytest.mark.parametrize(
    ("family", "task"),
    [("tabiclv2", "classification"), ("tabiclv2", "regression"), ("tabpfn35", "classification"), ("tabpfn35", "regression")],
)
def test_tabicl_and_tabpfn_adapters_configure_frozen_icl_context(
    family, task, monkeypatch, tmp_path
):
    checkpoint = tmp_path / "pretrained.safetensors"
    checkpoint.write_bytes(b"pretrained")
    calls = []

    class Estimator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model_ = torch.nn.Linear(1, 1)
            self.model_path_ = checkpoint
            self.model_path = checkpoint
            self.classes_ = np.array([1, 0])

        def fit(self, features, target):
            assert torch.is_inference_mode_enabled()
            calls.append(("fit", len(features), len(target)))

        def predict_proba(self, features):
            return np.tile([0.25, 0.75], (len(features), 1))

        def predict(self, features):
            return np.zeros(len(features), dtype=float)

    if family == "tabiclv2":
        class Native(Estimator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                calls.append(("constructor", kwargs))

        symbol = "TabICLClassifier" if task == "classification" else "TabICLRegressor"
        monkeypatch.setitem(sys.modules, "tabicl", SimpleNamespace(**{symbol: Native}))
    else:
        class Native:
            @classmethod
            def create_default_for_version(cls, model_version, **kwargs):
                calls.append(("constructor", model_version, kwargs))
                estimator = Estimator(**kwargs)
                estimator.models_ = [estimator.model_, torch.nn.Linear(1, 1)]
                return estimator

        monkeypatch.setitem(
            sys.modules,
            "tabpfn",
            SimpleNamespace(
                **{
                    "TabPFNClassifier" if task == "classification" else "TabPFNRegressor": Native
                }
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "tabpfn.constants",
            SimpleNamespace(ModelVersion=SimpleNamespace(V3_5="v3.5")),
        )

    suffix = "_regressor" if task == "regression" else ""
    model = create_model(_model_config(f"{family}_icl{suffix}", checkpoint))
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([1.0, 2.0, 3.0, 4.0]) if task == "regression" else pd.Series([0, 1, 0, 1])
    model.fit_context(features, target)

    constructor = calls[0][1:]
    if family == "tabiclv2":
        kwargs = constructor[0]
        assert kwargs["kv_cache"] == ("repr" if task == "regression" else "kv")
    else:
        assert constructor[0] == "v3.5"
        kwargs = constructor[1]
        assert kwargs["auto_scale_n_estimators"] is False
        assert kwargs["tuning_config"] is None
    assert kwargs["n_estimators"] == 8
    assert kwargs["device"] == "cuda:0"
    assert calls[-1] == ("fit", 4, 4)
    assert all(not module.training for module in model._modules())
    assert all(
        not parameter.requires_grad
        for module in model._modules()
        for parameter in module.parameters()
    )
    assert model.icl_metadata()["optimizer_steps"] == 0
    if task == "classification":
        np.testing.assert_allclose(model.predict_proba(features), [[0.75, 0.25]] * 4)
    else:
        np.testing.assert_allclose(model.predict(features), np.zeros(4))


def test_kumo_adapter_loads_pinned_weights_and_uses_icl_context(monkeypatch, tmp_path):
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.write_bytes(b"weights")

    class TableTensor:
        def __init__(self, frame):
            self.frame = frame

        @classmethod
        def from_pandas(cls, frame, **kwargs):
            return cls(frame)

        def __len__(self):
            return len(self.frame)

    class FakeKumo(torch.nn.Module):
        def __init__(self, task, size, pretrained, device):
            super().__init__()
            assert (task, size, pretrained, device) == (
                "classification",
                "medium",
                False,
                "cuda:0",
            )
            self.models = torch.nn.ModuleDict({task: torch.nn.Linear(1, 2)})
            self.fit_arguments = None

        def fit(self, x, y, *, num_estimators, generator):
            assert torch.is_inference_mode_enabled()
            self.fit_arguments = (len(x), len(y), num_estimators, generator)

        def predict(self, x):
            size = len(x)
            return {
                "0": SimpleNamespace(numerical=torch.full((size, 1), 0.8)),
                "1": SimpleNamespace(numerical=torch.full((size, 1), 0.2)),
            }

    monkeypatch.setitem(
        sys.modules,
        "sdm",
        SimpleNamespace(TableTensor=TableTensor, models=SimpleNamespace(KumoTabular=FakeKumo)),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kwargs: str(checkpoint)),
    )
    monkeypatch.setattr(FrozenICLModel, "_package_revision", staticmethod(lambda: PACKAGE_REVISION))
    monkeypatch.setattr(FrozenICLClassifier, "_generator", lambda self: "seeded")
    monkeypatch.setattr(torch.amp, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"weight": torch.ones((2, 1)), "bias": torch.zeros(2)},
    )

    model = create_model(_model_config("kumo_medium_icl", checkpoint))
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]})
    model.fit_context(features, pd.Series([0, 1, 0, 1]))

    assert model._estimator.fit_arguments == (4, 4, 8, "seeded")
    np.testing.assert_allclose(model.predict_proba(features), [[0.8, 0.2]] * 4)
    assert model.run_metadata()["checkpoint_sha256"] == model.icl_metadata()["checkpoint_sha256"]


def test_frozen_adapter_rejects_weight_changes_during_inference(tmp_path):
    checkpoint = tmp_path / "pretrained.ckpt"
    checkpoint.write_bytes(b"pretrained")

    class MutatingClassifier(FrozenICLClassifier):
        family = "tabiclv2"

        def _load_and_fit_context(self, features, target):
            class Estimator:
                model_ = torch.nn.Linear(1, 1)
                classes_ = np.array([0, 1])

                def predict_proba(self, features):
                    with torch.no_grad():
                        self.model_.weight.add_(1)
                    return np.tile([0.5, 0.5], (len(features), 1))

            self._estimator = Estimator()
            return checkpoint

    model = MutatingClassifier(_model_config("tabiclv2_icl", checkpoint))
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0]})
    model.fit_context(features, pd.Series([0, 1, 0, 1]))

    with pytest.raises(RuntimeError, match="weights changed"):
        model.predict_proba(features)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_repeated_icl_holdout_keeps_evaluation_rows_out_of_context(task):
    target = pd.Series(np.tile([0, 1], 40)) if task == "classification" else pd.Series(np.arange(80.0))
    participant_ids = pd.Series([f"p{index}" for index in range(len(target))])
    features = pd.DataFrame({"signal": target.astype(float), "varying": np.arange(len(target))})
    fitted_contexts = []

    class FakeModel:
        def fit_context(self, context_x, context_y):
            fitted_contexts.append((context_x["varying"].tolist(), len(context_y)))
            return self

        def predict_proba(self, evaluation_x):
            probability = np.clip(evaluation_x["signal"].to_numpy(dtype=float), 0.1, 0.9)
            return np.column_stack([1 - probability, probability])

        def predict(self, evaluation_x):
            return evaluation_x["signal"].to_numpy(dtype=float)

    metrics, assignments = repeated_icl_holdout(
        features,
        target,
        participant_ids,
        2,
        0.2,
        0.2,
        10,
        lambda seed: FakeModel(),
        task=task,
    )

    assert metrics["seed"].tolist() == [10, 11]
    assert len(fitted_contexts) == 2
    assert all(context_rows < len(target) for _, context_rows in fitted_contexts)
    for assignment in assignments.values():
        groups = {role: set(rows["participant_id"]) for role, rows in assignment.groupby("role")}
        assert set(groups) == {"icl_context", "reserved_validation", "evaluation"}
        assert not groups["icl_context"] & groups["evaluation"]
        assert not groups["reserved_validation"] & groups["evaluation"]
