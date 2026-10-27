import pandas as pd
import pytest

from daic_foundation_tab.tabular_study.folds import make_folds


def participants():
    return pd.DataFrame(
        [
            {
                "participant_id": str(index),
                "split": "train",
                "target_binary": index % 2,
                "target_phq8": index % 25,
            }
            for index in range(30)
        ]
        + [{"participant_id": "901", "split": "dev", "target_binary": 1, "target_phq8": 12}]
    )


def settings():
    return {
        "tasks": ["classification", "regression"],
        "folds": 5,
        "fold_seed": 42,
        "selection_fraction": 0.2,
        "selection_seed_offset": 10000,
    }


def test_folds_are_fixed_disjoint_and_evaluate_each_training_participant_once():
    data = participants()
    folds = make_folds(data, settings())
    assert folds == make_folds(data.sample(frac=1, random_state=9), settings())
    expected = set(data.loc[data["split"] == "train", "participant_id"])
    for task in folds:
        evaluation = []
        for fold in folds[task]["folds"]:
            fit, selection, held_out = (
                set(fold[role]) for role in ("fit", "selection", "evaluation")
            )
            assert not fit & selection and not fit & held_out and not selection & held_out
            assert fit | selection | held_out == expected
            evaluation.extend(held_out)
            if task == "classification":
                targets = data.set_index("participant_id")["target_binary"]
                assert all(
                    set(targets.loc[list(role)]) == {0, 1} for role in (fit, selection, held_out)
                )
        assert set(evaluation) == expected and len(evaluation) == len(expected)
        assert set(folds[task]["final"]["fit"]) | set(folds[task]["final"]["selection"]) == expected


def test_folds_reject_missing_class_in_evaluation():
    data = participants()
    data.loc[
        data["participant_id"].isin(
            ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26"]
        ),
        "target_binary",
    ] = 1
    with pytest.raises(ValueError):
        make_folds(data, settings())


def test_folds_accept_nullable_integer_labels_from_dataset():
    data = participants()
    data["target_binary"] = data["target_binary"].astype("Int64")
    assert len(make_folds(data, settings())["classification"]["folds"]) == 5
