from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from daic_foundation_tab.data.validation import fit_feature_selection
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.models.base import FineTunableClassifier


@dataclass(frozen=True)
class FineTunePartition:
    training_indices: np.ndarray
    validation_indices: np.ndarray
    assignment: pd.DataFrame


def stratified_fine_tune_partition(
    target: pd.Series,
    participant_ids: pd.Series,
    validation_fraction: float,
    random_state: int,
    *,
    training_role: str = "finetune_train",
    validation_role: str = "finetune_validation",
) -> FineTunePartition:
    if target.nunique() != 2:
        raise ValueError("Fine-tuning requires both target classes")
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=validation_fraction, random_state=random_state
        )
        training_indices, validation_indices = next(splitter.split(np.zeros(len(target)), target))
    except ValueError as error:
        raise ValueError("Unable to create a stratified fine-tuning validation split") from error
    training_target = target.iloc[training_indices]
    validation_target = target.iloc[validation_indices]
    if training_target.nunique() != 2 or validation_target.nunique() != 2:
        raise ValueError("Fine-tuning and early-stopping partitions must both contain two classes")
    assignment = pd.DataFrame(
        {
            "participant_id": participant_ids.iloc[np.r_[training_indices, validation_indices]].to_numpy(),
            "role": [training_role] * len(training_indices) + [validation_role] * len(validation_indices),
            "target": target.iloc[np.r_[training_indices, validation_indices]].to_numpy(),
            "random_state": random_state,
        }
    )
    return FineTunePartition(training_indices, validation_indices, assignment)


def _positive_probability(model: FineTunableClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier predict_proba must return two probability columns")
    return probabilities[:, 1]


def repeated_fine_tune_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    participant_ids: pd.Series,
    repeats: int,
    outer_validation_fraction: float,
    inner_validation_fraction: float,
    seed_start: int,
    model_factory: Callable[[int], FineTunableClassifier],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, float | int]] = []
    assignments: dict[int, pd.DataFrame] = {}
    for seed in range(seed_start, seed_start + repeats):
        outer = stratified_fine_tune_partition(
            target,
            participant_ids,
            outer_validation_fraction,
            seed,
            training_role="outer_train",
            validation_role="evaluation",
        )
        outer_features = features.iloc[outer.training_indices].reset_index(drop=True)
        outer_target = target.iloc[outer.training_indices].reset_index(drop=True)
        outer_ids = participant_ids.iloc[outer.training_indices].reset_index(drop=True)
        inner_seed = seed + 10_000
        inner = stratified_fine_tune_partition(
            outer_target,
            outer_ids,
            inner_validation_fraction,
            inner_seed,
        )
        selection = fit_feature_selection(outer_features.iloc[inner.training_indices])
        train_features = outer_features.iloc[inner.training_indices].reindex(columns=selection.columns)
        validation_features = outer_features.iloc[inner.validation_indices].reindex(columns=selection.columns)
        evaluation_features = features.iloc[outer.validation_indices].reindex(columns=selection.columns)
        model = model_factory(seed)
        with TemporaryDirectory(prefix="daic_ft_") as directory:
            model.fit(
                train_features,
                outer_target.iloc[inner.training_indices],
                validation_features=validation_features,
                validation_target=outer_target.iloc[inner.validation_indices],
                checkpoint_directory=Path(directory),
            )
            prediction = model.predict(evaluation_features)
            probability = _positive_probability(model, evaluation_features)
        metrics = classification_metrics(target.iloc[outer.validation_indices], prediction, probability)
        rows.append({"seed": seed, **{key: value for key, value in metrics.items() if isinstance(value, float)}})
        inner_assignment = inner.assignment.copy()
        inner_assignment["outer_seed"] = seed
        evaluation_assignment = outer.assignment.loc[outer.assignment["role"] == "evaluation"].copy()
        evaluation_assignment["outer_seed"] = seed
        assignments[seed] = pd.concat([inner_assignment, evaluation_assignment], ignore_index=True)
    return pd.DataFrame(rows), assignments
