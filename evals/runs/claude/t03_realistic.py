from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = [f"f{i}" for i in range(10)]


def _add_patient_features(data: pd.DataFrame) -> pd.DataFrame:
    """Label-free per-patient aggregates (uses only feature columns, never labels)."""
    df = data.copy()
    g = df.groupby("patient_id")[BASE]
    mean = g.transform("mean")
    std = g.transform("std").fillna(0.0)
    for c in BASE:
        df[f"{c}_pmean"] = mean[c]
        df[f"{c}_pstd"] = std[c]
        df[f"{c}_dev"] = df[c] - mean[c]
    df["n_visits"] = g[BASE[0]].transform("count")
    return df


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    df = _add_patient_features(data)
    features = [c for c in df.columns if c not in ("row_id", "patient_id", "label")]

    # Group-aware split: all visits of a patient land on the same side, so the
    # test score reflects generalization to unseen patients (no leakage).
    pat = df.groupby("patient_id")["label"].max().reset_index()
    train_p, test_p = train_test_split(
        pat["patient_id"],
        test_size=0.2,
        stratify=pat["label"],
        random_state=seed,
    )
    train = df[df["patient_id"].isin(set(train_p))]
    test = df[df["patient_id"].isin(set(test_p))]

    models = [
        RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            random_state=seed, n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            random_state=seed, n_jobs=1,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_depth=3,
            l2_regularization=1.0, random_state=seed,
        ),
    ]
    probs = []
    for m in models:
        m.fit(train[features], train["label"])
        probs.append(m.predict_proba(test[features])[:, 1])
    scores = np.mean(probs, axis=0)

    value = (
        float(roc_auc_score(test["label"], scores))
        if test["label"].nunique() > 1
        else float("nan")
    )

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": value,
        "notes": (
            "1) Replaced the random row split with a patient-level split "
            "(stratified by patient label): visits from the same patient were "
            "appearing in both train and test, which leaks patient identity and "
            "inflates the metric. The new number is an honest estimate for unseen "
            "patients and may be lower than before. "
            "2) Added label-free per-patient aggregates (mean, std, deviation "
            "from patient mean, visit count); this assumes a patient's other "
            "visits are available at prediction time. "
            "3) Replaced the single random forest with an average of RF, "
            "ExtraTrees and HistGradientBoosting with regularization "
            "(min_samples_leaf, shallow depth). Deterministic via seed."
        ),
    }