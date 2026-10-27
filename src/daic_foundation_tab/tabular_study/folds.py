from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from daic_foundation_tab.evaluation.fine_tuning import stratified_fine_tune_partition
from daic_foundation_tab.evaluation.regression_splits import regression_fine_tune_partition


def make_folds(table: pd.DataFrame, settings: dict) -> dict:
    train = table.loc[table["split"] == "train"].copy()
    train["participant_id"] = train["participant_id"].astype(str)
    train = train.sort_values("participant_id").reset_index(drop=True)
    if train.empty or train["participant_id"].duplicated().any():
        raise ValueError("Training participants must be nonempty and unique")
    ids = train["participant_id"]
    folds = {}
    for task in settings["tasks"]:
        target = train["target_binary" if task == "classification" else "target_phq8"]
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("Training targets must be finite")
        if task == "classification" and set(target.astype(int)) != {0, 1}:
            raise ValueError("Classification requires both classes")
        splitter = (
            StratifiedKFold(
                n_splits=settings["folds"], shuffle=True, random_state=settings["fold_seed"]
            )
            if task == "classification"
            else KFold(n_splits=settings["folds"], shuffle=True, random_state=settings["fold_seed"])
        )
        partitioner = (
            stratified_fine_tune_partition
            if task == "classification"
            else regression_fine_tune_partition
        )

        def partition(indices, seed):
            result = partitioner(
                target.iloc[indices].reset_index(drop=True),
                ids.iloc[indices].reset_index(drop=True),
                settings["selection_fraction"],
                seed,
                training_role="fit",
                validation_role="selection",
            )
            assignment = result.assignment
            return {
                role: sorted(
                    assignment.loc[assignment["role"] == role, "participant_id"].astype(str)
                )
                for role in ("fit", "selection")
            }

        task_folds = []
        for fold_index, (outer_train, outer_eval) in enumerate(splitter.split(train, target)):
            seed = settings["selection_seed_offset"] + settings["fold_seed"] + fold_index
            roles = partition(outer_train, seed)
            evaluation = sorted(ids.iloc[outer_eval].tolist())
            if task == "classification" and set(target.iloc[outer_eval].astype(int)) != {0, 1}:
                raise ValueError("Each classification evaluation fold must contain both classes")
            task_folds.append(
                {"fold": fold_index + 1, "selection_seed": seed, **roles, "evaluation": evaluation}
            )
        folds[task] = {
            "folds": task_folds,
            "final": {
                "selection_seed": settings["fold_seed"],
                **partition(np.arange(len(train)), settings["fold_seed"]),
            },
        }
    return folds
