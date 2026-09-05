from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daic_foundation_tab.data.aggregation import COVAREP_COLUMNS


def experiment_config(dataset_root: Path, cache_root: Path) -> dict:
    return {
        "project": {"seed": 42, "cache_root": str(cache_root), "output_root": str(cache_root / "outputs")},
        "data": {
            "root": str(dataset_root),
            "labels": {
                "train": "original_labels/train.csv",
                "dev": "original_labels/dev.csv",
                "test": "original_labels/test.csv",
            },
            "missing_modality_policy": "error",
            "source_groups": {
                "covarep": {
                    "modality": "audio",
                    "parser": "covarep",
                    "pattern": "data/{participant_id}_COVAREP.csv",
                },
                "action_units": {
                    "modality": "visual",
                    "parser": "openface",
                    "pattern": "data/{participant_id}_CLNF_AUs.txt",
                },
                "gaze": {
                    "modality": "visual",
                    "parser": "openface",
                    "pattern": "data/{participant_id}_CLNF_gaze.txt",
                },
                "head_pose": {
                    "modality": "visual",
                    "parser": "openface",
                    "pattern": "data/{participant_id}_CLNF_pose.txt",
                },
            },
        },
        "modalities": {
            "audio": {"enabled": True, "groups": ["covarep"]},
            "visual": {"enabled": True, "groups": ["action_units", "gaze", "head_pose"]},
            "text_numeric": {"enabled": False, "groups": []},
        },
        "aggregation": {"statistics": ["mean", "std"], "std_ddof": 0},
        "experiment": {"task": "classification", "feature_set": "audio_visual"},
        "model": {
            "name": "tabiclv2_ft",
            "checkpoint_version": "test",
            "parameters": {"eval_metric": "roc_auc", "random_state": 42},
        },
        "evaluation": {
            "threshold": 0.5,
            "test_predictions": False,
            "fine_tuning": {"validation_fraction": 0.2, "validation_seed": 42},
            "repeated_holdout": {"enabled": False},
        },
        "bootstrap": {"enabled": False},
        "logging": {"level": "INFO"},
    }


def write_synthetic_dataset(root: Path) -> dict:
    (root / "data").mkdir(parents=True)
    (root / "original_labels").mkdir()
    train_ids = ["300", "301", "302", "303", "304", "305"]
    dev_ids = ["400", "401"]
    test_ids = ["500", "501"]
    pd.DataFrame(
        {
            "Participant_ID": train_ids,
            "PHQ8_Binary": [0, 1, 0, 1, 0, 1],
            "PHQ8_Score": [2, 11, 3, 12, 4, 13],
        }
    ).to_csv(root / "original_labels/train.csv", index=False)
    pd.DataFrame(
        {
            "Participant_ID": dev_ids,
            "PHQ8_Binary": [0, 1],
            "PHQ8_Score": [5, 14],
        }
    ).to_csv(root / "original_labels/dev.csv", index=False)
    pd.DataFrame({"participant_ID": test_ids, "Gender": [0, 1]}).to_csv(
        root / "original_labels/test.csv", index=False
    )

    for participant_id in train_ids + dev_ids + test_ids:
        value = int(participant_id) / 100
        covarep = np.tile(np.arange(len(COVAREP_COLUMNS), dtype=float), (3, 1)) + value
        covarep[:, 0] = [0.0, 0.01, 0.02]
        pd.DataFrame(covarep).to_csv(
            root / "data" / f"{participant_id}_COVAREP.csv", header=False, index=False
        )
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "timestamp": [0.0, 0.03, 0.06],
                "confidence": [1.0, 1.0, 1.0],
                "success": [1, 1, 1],
                "AU01_r": [value, value + 1, value + 2],
            }
        ).to_csv(root / "data" / f"{participant_id}_CLNF_AUs.txt", index=False)
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "timestamp": [0.0, 0.03, 0.06],
                "confidence": [1.0, 1.0, 1.0],
                "success": [1, 1, 1],
                "x_0": [value, value + 1, value + 2],
            }
        ).to_csv(root / "data" / f"{participant_id}_CLNF_gaze.txt", index=False)
        pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "timestamp": [0.0, 0.03, 0.06],
                "confidence": [1.0, 1.0, 1.0],
                "success": [1, 1, 1],
                "Tx": [value, value + 1, value + 2],
            }
        ).to_csv(root / "data" / f"{participant_id}_CLNF_pose.txt", index=False)
    return experiment_config(root, root / "cache")
