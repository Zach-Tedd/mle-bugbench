
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit


FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------
    df = data.copy()

    X = df[FEATURES].astype(float)
    y = df["label"].astype(int)

    if "row_id" not in df.columns:
        raise ValueError("Input data must contain a row_id column.")

    if y.nunique() != 2:
        raise ValueError("Both classes must be present.")

    # ---------------------------------------------------------
    # 2. Identify duplicate observations
    # ---------------------------------------------------------
    # Identical feature vectors must not be distributed across
    # training and testing, otherwise evaluation is optimistic.

    group_ids = (
        df.groupby(FEATURES, sort=False, dropna=False)
        .ngroup()
        .to_numpy()
    )

    unique_groups = np.unique(group_ids)

    # Every duplicate group has the same label in this dataset.
    group_labels = (
        df.drop_duplicates(FEATURES)
        .set_index(FEATURES)["label"]
    )

    unique_data = (
        df.drop_duplicates(FEATURES)
        .reset_index(drop=True)
    )

    unique_y = unique_data["label"].to_numpy()

    # ---------------------------------------------------------
    # 3. Stratified group-level train/test split
    # ---------------------------------------------------------
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_group_idx, test_group_idx = next(
        splitter.split(
            unique_data[FEATURES],
            unique_y,
        )
    )

    train_groups = unique_groups[train_group_idx]
    test_groups = unique_groups[test_group_idx]

    train_mask = np.isin(group_ids, train_groups)
    test_mask = np.isin(group_ids, test_groups)

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    X_train = train[FEATURES]
    y_train = train["label"].astype(int)

    X_test = test[FEATURES]
    y_test = test["label"].astype(int)

    # ---------------------------------------------------------
    # 4. Random Forest
    # ---------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features=0.8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    rf_scores = rf.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 5. Extra Trees
    # ---------------------------------------------------------
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        bootstrap=False,
        random_state=seed + 1,
        n_jobs=-1,
    )

    et.fit(X_train, y_train)

    et_scores = et.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 6. Ensemble predictions
    # ---------------------------------------------------------
    # Averaging two independently randomized tree ensembles
    # generally produces more stable predictions.

    scores = (
        0.5 * rf_scores +
        0.5 * et_scores
    )

    scores = np.clip(scores, 0.0, 1.0)

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------
    value = average_precision_score(
        y_test,
        scores,
    )

    # ---------------------------------------------------------
    # 8. Return results
    # ---------------------------------------------------------
    return {
        "train_ids": train["row_id"].astype(int).tolist(),

        "test_ids": test["row_id"].astype(int).tolist(),

        "test_scores": scores.astype(float).tolist(),

        "metric": "average_precision",

        "value": float(value),

        "notes": (
            "Improved fraud detection using a stratified "
            "group-level train/test split to prevent identical "
            "feature vectors from leaking across partitions. "
            "Preserved the natural class distribution and used "
            "class-weighted Random Forest and Extra Trees "
            "classifiers. Combined their predicted probabilities "
            "using equal-weight averaging. Evaluated using "
            "Average Precision (PR-AUC), which is more suitable "
            "than accuracy for imbalanced fraud detection. "
            "All random operations are controlled by the seed."
        ),
    }
