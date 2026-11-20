"""Biomarker model with leakage-free cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"g{i}" for i in range(2000)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    row_ids = data["row_id"].to_numpy()

    folds = []
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        # Feature selection is part of the pipeline, so it is fit on the
        # training fold only and never sees validation labels.
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=50)),
                ("clf", LogisticRegression(C=0.1, max_iter=1000)),
            ]
        )
        model.fit(x[train_idx], y[train_idx])
        scores = model.predict_proba(x[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], scores))
        folds.append(
            {
                "train_ids": row_ids[train_idx].tolist(),
                "val_ids": row_ids[val_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Problem: SelectKBest was fit on all 200 samples (including labels) before "
        "cross-validation. With 2000 noisy features and n=200, this picks genes that "
        "correlate with the label by chance, and the validation folds had already "
        "influenced that choice, so the reported AUC of 0.94 was badly optimistic "
        "(selection leakage). Fix: scaling, feature selection (top 50 by ANOVA F) and "
        "logistic regression are now one Pipeline refit inside each training fold, so "
        "validation samples never affect selection or fitting. I also added L2 "
        "regularisation (C=0.1) to suit the small-n, high-dimensional setting. "
        "Stratified 5-fold CV with a seeded shuffle keeps results deterministic, and "
        "every sample is validated exactly once. The value is the mean per-fold ROC-AUC "
        "and is expected to be far lower than 0.94, since only ~10 genes carry a weak "
        "signal."
    )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }