from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import psutil

from daic_foundation_tab.data.validation import fit_feature_selection
from daic_foundation_tab.evaluation.fine_tuning import stratified_fine_tune_partition
from daic_foundation_tab.evaluation.metrics import classification_metrics
from daic_foundation_tab.evaluation.regression import regression_metrics
from daic_foundation_tab.evaluation.regression_splits import nested_regression_partitions
from daic_foundation_tab.models.icl import FrozenICLModel


def repeated_icl_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    participant_ids: pd.Series,
    repeats: int,
    outer_fraction: float,
    inner_fraction: float,
    seed_start: int,
    model_factory: Callable[[int], FrozenICLModel],
    *,
    task: str,
    threshold: float = 0.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    if repeats < 1:
        raise ValueError("Repeated frozen ICL holdout requires at least one repeat")
    rows = []
    assignments = {}
    logger = logging.getLogger(__name__)
    process = psutil.Process()
    for completed, seed in enumerate(range(seed_start, seed_start + repeats), start=1):
        target = target.reset_index(drop=True)
        participant_ids = participant_ids.reset_index(drop=True)
        if task == "regression":
            nested = nested_regression_partitions(
                target, participant_ids, outer_fraction, inner_fraction, seed
            )
            outer, inner = nested.outer, nested.inner
            outer_features = features.iloc[outer.training_indices].reset_index(drop=True)
            outer_target = target.iloc[outer.training_indices].reset_index(drop=True)
            evaluation_indices = outer.validation_indices
            assignment = nested.assignment.copy()
        else:
            outer = stratified_fine_tune_partition(
                target,
                participant_ids,
                outer_fraction,
                seed,
                training_role="outer_train",
                validation_role="evaluation",
            )
            outer_features = features.iloc[outer.training_indices].reset_index(drop=True)
            outer_target = target.iloc[outer.training_indices].reset_index(drop=True)
            outer_ids = participant_ids.iloc[outer.training_indices].reset_index(drop=True)
            inner = stratified_fine_tune_partition(
                outer_target,
                outer_ids,
                inner_fraction,
                seed + 10_000,
            )
            assignment = pd.concat(
                [
                    inner.assignment,
                    outer.assignment.loc[outer.assignment["role"] == "evaluation"],
                ],
                ignore_index=True,
            )
            evaluation_indices = outer.validation_indices

        selection = fit_feature_selection(outer_features.iloc[inner.training_indices])
        context_x = outer_features.iloc[inner.training_indices].reindex(columns=selection.columns)
        context_y = outer_target.iloc[inner.training_indices]
        evaluation_x = features.iloc[evaluation_indices].reindex(columns=selection.columns)
        model = model_factory(seed)
        try:
            logger.info(
                "phase=preparing repeated ICL context seed=%s progress=%s/%s repeats rss_mib=%.1f",
                seed, completed - 1, repeats, process.memory_info().rss / 1024**2,
            )
            model.fit_context(context_x, context_y)
            logger.info(
                "phase=predicting repeated ICL holdout seed=%s rss_mib=%.1f",
                seed, process.memory_info().rss / 1024**2,
            )
            if task == "regression":
                prediction = np.asarray(model.predict(evaluation_x), dtype=float)
                metrics = regression_metrics(target.iloc[evaluation_indices], prediction)
            else:
                probability = np.asarray(model.predict_proba(evaluation_x), dtype=float)[:, 1]
                prediction = (probability > threshold).astype(int)
                metrics = classification_metrics(target.iloc[evaluation_indices], prediction, probability)
        finally:
            model.close()
            logger.info(
                "phase=releasing repeated ICL context seed=%s rss_mib=%.1f",
                seed, process.memory_info().rss / 1024**2,
            )
        rows.append(
            {"seed": seed, **{key: value for key, value in metrics.items() if isinstance(value, float)}}
        )
        assignment["role"] = assignment["role"].replace(
            {"finetune_train": "icl_context", "finetune_validation": "reserved_validation"}
        )
        assignment["outer_seed"] = seed
        assignments[seed] = assignment
        if progress_callback is not None:
            progress_callback(completed, repeats)
    return pd.DataFrame(rows), assignments
