from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd

from daic_foundation_tab.data.dataset import ParticipantDataset
from daic_foundation_tab.evaluation.fine_tuning import (
    prepare_fine_tune_splits,
    repeated_fine_tune_holdout,
    stratified_fine_tune_partition,
)
from daic_foundation_tab.models.tabiclv2_ft import TabICLv2FineTunedModel


class _FakeFineTuner:
    instances: ClassVar[list[_FakeFineTuner]] = []

    def __init__(self, **parameters) -> None:
        self.parameters = parameters
        self._best_metric_ = 0.75
        self.fit_arguments = None
        self.instances.append(self)

    def fit(self, features, target, X_val, y_val, output_dir) -> None:
        self.fit_arguments = (features.copy(), target.copy(), X_val.copy(), y_val.copy())
        (output_dir / "best.ckpt").write_bytes(b"checkpoint")

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.7, 0.3], (len(features), 1))


class _FakeFineTunedModel:
    def fit(self, features, target, *, validation_features, validation_target, checkpoint_directory):
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        (checkpoint_directory / "best.ckpt").write_bytes(b"checkpoint")
        return self

    def predict(self, features):
        return np.zeros(len(features), dtype=int)

    def predict_proba(self, features):
        return np.tile([0.7, 0.3], (len(features), 1))

    def validate_input(self, features):
        return None

    def run_metadata(self):
        return {}

    def finetune_metadata(self):
        return {}


def test_fine_tuned_adapter_passes_explicit_validation_data(monkeypatch, tmp_path) -> None:
    _FakeFineTuner.instances.clear()
    monkeypatch.setitem(sys.modules, "tabicl", SimpleNamespace(FinetunedTabICLClassifier=_FakeFineTuner))
    model = TabICLv2FineTunedModel(
        {
            "checkpoint_version": "checkpoint.ckpt",
            "parameters": {"eval_metric": "roc_auc", "amp": False, "device": "cpu"},
        }
    )
    features = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 0, 1])

    model.fit(
        features.iloc[:2],
        target.iloc[:2],
        validation_features=features.iloc[2:],
        validation_target=target.iloc[2:],
        checkpoint_directory=tmp_path / "checkpoints",
    )

    instance = _FakeFineTuner.instances[0]
    assert instance.parameters["checkpoint_version"] == "checkpoint.ckpt"
    assert instance.fit_arguments[2].equals(features.iloc[2:])
    assert model.finetune_metadata()["best_validation_metric"] == 0.75


def test_stratified_fine_tune_partition_is_reproducible_and_disjoint() -> None:
    target = pd.Series([0, 1] * 10)
    participant_ids = pd.Series([str(index) for index in range(len(target))])

    first = stratified_fine_tune_partition(target, participant_ids, 0.2, 42)
    second = stratified_fine_tune_partition(target, participant_ids, 0.2, 42)

    assert first.assignment.equals(second.assignment)
    assert not set(first.training_indices) & set(first.validation_indices)
    assert target.iloc[first.training_indices].nunique() == 2
    assert target.iloc[first.validation_indices].nunique() == 2


def test_repeated_fine_tune_holdout_nests_disjoint_partitions() -> None:
    target = pd.Series([0, 1] * 10)
    participant_ids = pd.Series([str(index) for index in range(len(target))])
    features = pd.DataFrame({"feature": np.arange(len(target), dtype=float)})

    metrics, assignments = repeated_fine_tune_holdout(
        features,
        target,
        participant_ids,
        repeats=2,
        outer_validation_fraction=0.2,
        inner_validation_fraction=0.25,
        seed_start=0,
        model_factory=lambda _: _FakeFineTunedModel(),
    )

    assert metrics["seed"].tolist() == [0, 1]
    for assignment in assignments.values():
        groups = {role: set(group["participant_id"]) for role, group in assignment.groupby("role")}
        assert not groups["finetune_train"] & groups["finetune_validation"]
        assert not groups["finetune_train"] & groups["evaluation"]
        assert not groups["finetune_validation"] & groups["evaluation"]


def test_fine_tune_feature_selection_uses_only_inner_training_partition() -> None:
    target = pd.Series([0, 1] * 5)
    participant_ids = pd.Series([str(index) for index in range(len(target))])
    partition = stratified_fine_tune_partition(target, participant_ids, 0.2, 42)
    inner_constant = np.zeros(len(target), dtype=float)
    inner_constant[partition.validation_indices] = 1.0
    table = pd.DataFrame(
        {
            "participant_id": [*participant_ids, "dev_0", "dev_1", "test_0"],
            "split": ["train"] * len(target) + ["dev", "dev", "test"],
            "target_binary": [*target, 0, 1, pd.NA],
            "target_phq8": [*target, 0, 10, np.nan],
            "signal": [*np.arange(len(target), dtype=float), 20.0, 21.0, 22.0],
            "inner_constant": [*inner_constant, 99.0, 100.0, 101.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "column": ["signal", "inner_constant"],
            "modality": ["audio", "audio"],
            "source_group": ["test", "test"],
            "source_feature": ["signal", "inner_constant"],
            "aggregation": ["mean", "mean"],
        }
    )
    dataset = ParticipantDataset(table, manifest, pd.DataFrame(), {}, {}, "test")

    prepared = prepare_fine_tune_splits(dataset, "audio", 0.2, 42)

    assert "inner_constant" not in prepared.selection.columns
    assert list(prepared.dev_x.columns) == list(prepared.training_x.columns)
