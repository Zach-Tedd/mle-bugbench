"""Reference fix: select genes only within each cross-validation training fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

FEATURES = [f"g{i}" for i in range(2000)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    folds = []
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for train_idx, val_idx in splitter.split(x, y):
        model = make_pipeline(
            SelectKBest(f_classif, k=50),
            LogisticRegression(max_iter=1000),
        )
        model.fit(x[train_idx], y[train_idx])
        scores = model.predict_proba(x[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], scores))
        folds.append(
            {
                "train_ids": data["row_id"].iloc[train_idx].tolist(),
                "val_ids": data["row_id"].iloc[val_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "The starter selected genes once using all labels, including validation labels, "
            "before cross-validation. Feature selection now runs inside each fold's pipeline. "
            "The leaky starter still scores about 0.91 ROC-AUC on shuffled labels, while honest "
            "selection should be near chance; the real-label cross-validation AUC is about 0.73."
        ),
    }
