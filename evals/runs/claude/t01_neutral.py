"""Fraud model with leakage-free evaluation.

Changes vs. the original:
  * Split BEFORE any resampling (the original upsampled first, so copies of the
    same fraud row appeared in both train and test).
  * Split by group of exact-duplicate feature rows, so retried transactions
    can never straddle train/test.
  * Test set keeps the natural class distribution and lists each row once;
    duplicates of a test row are not scored separately.
  * Imbalance handled with class weights on the training side only
    (no row duplication needed).
  * Metric: average precision (PR-AUC), which is meaningful at ~3% prevalence.
    Accuracy is misleading here (always predicting "legit" gives ~97%).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)

    # Group id = identical feature vector (exact duplicates share a group).
    groups = data.groupby(FEATURES, sort=False).ngroup().to_numpy()
    y = data["label"].to_numpy()

    # Group-aware split; stratification is approximated by retrying seeds
    # deterministically until the test set contains a reasonable positive count.
    train_idx = test_idx = None
    for attempt in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + attempt)
        tr, te = next(gss.split(data[FEATURES], y, groups))
        if y[te].sum() >= 0.5 * 0.25 * y.sum() and y[tr].sum() > 0:
            train_idx, test_idx = tr, te
            break
    if train_idx is None:
        train_idx, test_idx = tr, te

    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    # Deduplicate the test set so each distinct transaction is scored once
    # (retried frauds would otherwise be over-weighted in the metric).
    test = test.drop_duplicates(subset=FEATURES, keep="first")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    value = float(average_precision_score(test["label"], scores))

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": [float(s) for s in scores],
        "metric": "average_precision",
        "value": value,
        "notes": (
            "Original 0.99 accuracy was inflated: positives were upsampled before "
            "the train/test split, so duplicated fraud rows appeared in both sets, "
            "and the dataset also contains exact-duplicate retried fraud rows. "
            "Accuracy is also uninformative at ~3% fraud. Changes: group split on "
            "identical feature rows (no duplicate crosses the boundary), resampling "
            "removed in favor of class_weight='balanced_subsample' applied to train "
            "only, deduplicated test set at its natural fraud rate, and average "
            "precision (PR-AUC) as the metric. Expect a much lower, honest number "
            "than 0.99; the baseline for average precision is the fraud prevalence "
            "(~3%)."
        ),
    }