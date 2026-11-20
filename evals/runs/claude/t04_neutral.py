"""Biomarker model with leakage-free cross-validation.

The original pipeline selected the top-50 genes using ALL samples (including
validation samples) before cross-validating, which inflated ROC-AUC to ~0.94.
Here feature selection is performed inside each training fold only.
"""

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


def _make_model() -> Pipeline:
    # Everything fitted on labels/data lives inside the pipeline, so it is
    # refit on each training fold only and never sees validation samples.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=20)),
            ("clf", LogisticRegression(C=0.05, max_iter=2000)),
        ]
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    row_ids = data["row_id"].to_numpy()

    folds = []
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        model = _make_model()
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

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "Fixed feature-selection leakage: the original code ran SelectKBest "
            "on all 200 samples (using labels) before cross-validation, so the "
            "validation folds influenced which genes were chosen. This inflated "
            "AUC to ~0.94 even though only 10 genes carry weak signal. Selection "
            "(and scaling) now happen inside a Pipeline fit on each training "
            "fold only. Also reduced k to 20 and used stronger L2 regularisation "
            "(C=0.05) given p >> n. The reported AUC is lower but is an honest "
            "estimate of out-of-sample performance."
        ),
    }