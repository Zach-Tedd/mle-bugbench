"""Crop-disease classifier evaluated on geographically unseen counties.

Uses leave-one-county-out cross-validation to estimate generalization
to counties not represented in the training data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# f0-f2 contain the disease-related signal shared across counties.
# f3 contains a known Nyeri-specific spurious shade cue.
# Exclude f3 to prevent the model from learning a geographically
# specific correlation that does not generalize.
FEATURES = ["f0", "f1", "f2"]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Train and evaluate using leave-one-county-out cross-validation.

    Each fold holds out one entire county. The model is trained using
    samples from all remaining counties and evaluated on the unseen
    county.

    Returns:
        folds: Training IDs, validation IDs and predicted probabilities.
        metric: Evaluation metric name.
        value: Mean ROC-AUC across the held-out counties.
        notes: Explanation of the evaluation and modeling changes.
    """

    regions = sorted(data["region"].dropna().unique().tolist())

    if len(regions) < 2:
        raise ValueError("At least two distinct counties are required.")

    missing = [
        column
        for column in ["row_id", "region", "label", *FEATURES]
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique.")

    if data[["region", "label", *FEATURES]].isna().any().any():
        raise ValueError("Required data columns contain missing values.")

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0 or 1).")

    folds = []
    fold_aucs = []

    for region in regions:

        # Geographic holdout: no samples from the validation county
        # are permitted in the training data.
        val_mask = data["region"] == region
        train = data.loc[~val_mask].copy()
        val = data.loc[val_mask].copy()

        if train.empty or val.empty:
            raise ValueError(f"Invalid training or validation fold: {region}")

        if train["label"].nunique() != 2:
            raise ValueError(
                f"Training fold excluding {region} does not contain both classes."
            )

        if val["label"].nunique() != 2:
            raise ValueError(
                f"Validation county {region} does not contain both classes."
            )

        # Scaling is fitted exclusively on the training counties.
        # Logistic regression provides a regularized model that learns
        # the shared disease signal without relying on the Nyeri cue.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        )

        model.fit(train[FEATURES], train["label"])

        # Probability of disease (positive class).
        scores = model.predict_proba(val[FEATURES])[:, 1]

        auc = roc_auc_score(val["label"], scores)
        fold_aucs.append(float(auc))

        folds.append(
            {
                "train_ids": train["row_id"].tolist(),
                "val_ids": val["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(fold_aucs)),
        "notes": (
            "The original evaluation randomly split Nyeri samples into "
            "training and validation sets. This measured performance on "
            "the same county rather than generalization to unseen counties. "
            "It also allowed the model to exploit f3, a known spurious shade "
            "cue specific to Nyeri, potentially inflating the reported "
            "ROC-AUC of 0.97. Replaced the random split with five-fold "
            "leave-one-county-out cross-validation. Each fold trains on "
            "four counties and validates on the entirely unseen fifth "
            "county. Excluded f3 because its disease association is "
            "geographically specific rather than a reliable shared "
            "disease signal. Retained f0-f2, which contain the shared "
            "disease-related information, and used standardized, "
            "L2-regularized logistic regression. Scaling is fitted "
            "independently within each training fold to prevent leakage. "
            "The reported value is the unweighted mean of the five "
            "held-out-county ROC-AUC scores. This estimates geographic "
            "generalization across the five sampled counties, not "
            "guaranteed performance across all Kenyan counties. "
            "Further external validation on independently collected "
            "farms and additional counties is necessary before national "
            "deployment."
        ),
    }