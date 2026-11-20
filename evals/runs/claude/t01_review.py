"""Fraud model with leakage fixed and an honest evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)

    # Group exact-duplicate rows (same features and label) so that copies
    # of the same transaction can never land on both sides of the split.
    group = data.groupby(FEATURES + ["label"], sort=False).ngroup().to_numpy()
    data = data.assign(_group=group)

    # One row per group for deciding the split (labels are constant within a group).
    groups = (
        data.groupby("_group", sort=True)
        .agg(label=("label", "first"))
        .reset_index()
    )

    # Stratified split at the group level, deterministic for a given seed.
    rng = np.random.default_rng(seed)
    test_groups: set[int] = set()
    for lab in (0, 1):
        g = groups.loc[groups["label"] == lab, "_group"].to_numpy()
        g = rng.permutation(g)
        n_test = int(round(0.25 * len(g)))
        test_groups.update(g[:n_test].tolist())

    is_test = data["_group"].isin(test_groups).to_numpy()
    train = data[~is_test]
    test_all = data[is_test]

    # Evaluate each distinct test transaction once (natural prevalence,
    # no resampling, duplicates collapsed so retries don't inflate the score).
    test = test_all.drop_duplicates(subset="_group", keep="first")

    # Handle imbalance with class weights rather than resampling.
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    value = float(average_precision_score(test["label"], scores))

    notes = (
        "Problems found: (1) minority class was upsampled BEFORE train/test "
        "split, so duplicated positives leaked into the test set; (2) the data "
        "contains exact-duplicate rows (fraud retries) that also leak across "
        "the split; (3) the test set was artificially balanced, not reflecting "
        "the ~3% real fraud rate; (4) accuracy is misleading at 3% prevalence "
        "(always predicting 'legit' gives ~0.97). Changes: grouped exact "
        "duplicates and did a stratified split at the group level so no "
        "duplicate straddles train/test; no resampling, using "
        "class_weight='balanced_subsample' instead; test set keeps the natural "
        "class balance with each distinct transaction scored once; reported "
        "average precision (PR-AUC), which is meaningful for rare positives. "
        "The honest value is far below the claimed 0.99."
    )

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": [float(s) for s in scores],
        "metric": "average_precision",
        "value": value,
        "notes": notes,
    }