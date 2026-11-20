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
    """Scaler -> univariate selection -> L2 logistic regression.

    All steps are fit only on the data passed to .fit(), so when this object
    is fit on a training fold, validation rows never influence scaling,
    feature selection, or hyperparameters. The grid search runs an inner CV
    on the training fold to pick the number of features and regularization.
    """
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    grid = {
        "select__k": [10, 25, 50, 100, 250],
        "clf__C": [0.001, 0.01, 0.1, 1.0],
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(
        pipe,
        grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score=0.0,
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy()
    row_ids = data["row_id"].to_numpy()

    folds = []
    aucs = []
    chosen = []

    # Outer split: every sample is validated exactly once.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        search = _build_search(seed)
        search.fit(x[train_idx], y[train_idx])  # selection + tuning on train only
        scores = search.predict_proba(x[val_idx])[:, 1]

        aucs.append(roc_auc_score(y[val_idx], scores))
        chosen.append(search.best_params_)
        folds.append(
            {
                "train_ids": row_ids[train_idx].tolist(),
                "val_ids": row_ids[val_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Fixed leakage: the original fit SelectKBest on all 200 rows (including "
        "validation labels) before cross-validation, which with 2000 features "
        "and 200 samples gives an optimistically biased AUC. Now feature "
        "selection and scaling happen inside each training fold via a Pipeline. "
        "Added StandardScaler, class_weight='balanced', and an inner "
        "5-fold GridSearchCV (on the training fold only) over the number of "
        "selected features k in {10,25,50,100,250} and L2 strength C in "
        "{0.001,0.01,0.1,1}, since the original k=50 and default C were "
        "untuned for the p>>n regime. Fully deterministic given seed. "
        "Expect the reported AUC to be lower than the original's, because it "
        "is now an honest estimate. Chosen params per fold: "
        + "; ".join(str(p) for p in chosen)
    )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }