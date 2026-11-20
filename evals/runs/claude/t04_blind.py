from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"g{i}" for i in range(2000)]


def _build_search(seed: int) -> GridSearchCV:
    # All steps (scaling, selection, classifier) are fit only on the data the
    # pipeline is given, so nothing from validation rows can leak in.
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    grid = {
        "select__k": [10, 20, 50, 100],
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0],
    }
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    return GridSearchCV(
        pipe,
        grid,
        scoring="roc_auc",
        cv=inner,
        n_jobs=1,
        refit=True,
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy()
    row_ids = data["row_id"].to_numpy()

    folds = []
    aucs = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in outer.split(x, y):
        # Hyperparameters and feature selection are tuned/fit on the training
        # fold only (nested CV); the validation fold is used solely for scoring.
        search = _build_search(seed)
        search.fit(x[train_idx], y[train_idx])
        scores = search.predict_proba(x[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], scores))
        folds.append(
            {
                "train_ids": row_ids[train_idx].tolist(),
                "val_ids": row_ids[val_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Fixed leakage: the original fit SelectKBest on all samples (including "
        "validation labels) before cross-validation, inflating AUC. Now "
        "StandardScaler + SelectKBest + LogisticRegression run in a Pipeline "
        "fit only on each training fold. Added nested CV (inner 5-fold "
        "GridSearchCV on the training fold) to tune the number of selected "
        "features (10-100) and L2 strength C, since the best k/C is "
        "unknown a priori with 2000 noisy features and ~160 training samples. "
        "The reported roc_auc is the mean over outer folds and is an honest, "
        "unbiased estimate; it will be lower than the leaky original number."
    )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }