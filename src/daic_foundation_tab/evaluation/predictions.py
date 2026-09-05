from __future__ import annotations

import numpy as np
import pandas as pd


def classification_predictions(
    participant_ids: pd.Series,
    split: str,
    prediction: object,
    probability_positive: object,
    target: pd.Series | None = None,
) -> pd.DataFrame:
    probabilities = np.asarray(probability_positive, dtype=float)
    frame = pd.DataFrame(
        {
            "participant_id": participant_ids.astype(str).to_numpy(),
            "split": split,
            "y_true": target.to_numpy() if target is not None else pd.NA,
            "y_pred": np.asarray(prediction, dtype=int),
            "prob_non_depressed": 1 - probabilities,
            "prob_depressed": probabilities,
        }
    )
    return frame
