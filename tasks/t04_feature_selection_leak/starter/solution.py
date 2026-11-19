"""Biomarker model. The analyst reports cross-validated ROC-AUC 0.94 and wants to publish."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

FEATURES = [f"g{i}" for i in range(2000)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES].to_numpy()
    y = data["label"].to_numpy()
    selector = SelectKBest(f_classif, k=50)
    x_selected = selector.fit_transform(x, y)

    folds = []
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x_selected, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(x_selected[train_idx], y[train_idx])
        scores = model.predict_proba(x_selected[val_idx])[:, 1]
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
        "notes": "",
    }
