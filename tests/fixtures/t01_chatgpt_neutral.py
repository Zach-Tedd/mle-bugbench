
"""Leakage-resistant fraud detection model.

Improvements:
- Prevents exact duplicate rows from crossing train/test boundaries.
- Preserves the natural fraud prevalence.
- Uses Average Precision (PR-AUC) instead of accuracy.
- Uses a regularized Random Forest.
- Deterministic for a given seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # Work with a copy to avoid modifying the caller's DataFrame.
    data = data.copy()

    # Identify exact duplicate feature vectors.
    # All identical feature vectors must belong to the same split.
    group_ids = (
        data.groupby(
            FEATURES,
            dropna=False,
            sort=False
        ).ngroup()
    )

    data["_group_id"] = group_ids.to_numpy()

    # Build a table of unique feature groups.
    groups = (
        data[["_group_id", "label"]]
        .drop_duplicates("_group_id")
        .reset_index(drop=True)
    )

    # Split unique groups, stratifying by their labels.
    # This prevents duplicate-feature leakage between train and test.
    train_groups, test_groups = train_test_split(
        groups["_group_id"].to_numpy(),
        test_size=0.2,
        random_state=seed,
        stratify=groups["label"].to_numpy(),
    )

    # Recover original rows belonging to each partition.
    train = data[data["_group_id"].isin(train_groups)].copy()
    test = data[data["_group_id"].isin(test_groups)].copy()

    # Train on the original distribution.
    # Do not oversample before splitting.
    X_train = train[FEATURES]
    y_train = train["label"]

    X_test = test[FEATURES]
    y_test = test["label"]

    # Regularized Random Forest.
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Fraud probabilities.
    scores = model.predict_proba(X_test)[:, 1]

    # Average Precision is more informative than accuracy
    # for highly imbalanced fraud detection problems.
    value = average_precision_score(y_test, scores)

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": scores.astype(float).tolist(),
        "metric": "average_precision",
        "value": float(value),
        "notes": (
            "Replaced the original oversample-before-split approach "
            "with a group-based train/test split. Exact duplicate "
            "feature vectors are kept entirely within one partition "
            "to prevent leakage. Preserved the natural class "
            "distribution without artificial oversampling. Used a "
            "500-tree regularized Random Forest with a fixed random "
            "seed. Replaced accuracy with Average Precision (PR-AUC), "
            "which better evaluates fraud detection under severe "
            "class imbalance. Test probabilities are generated only "
            "for previously unseen feature groups."
        ),
    }
