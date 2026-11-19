"""Reference fix: validate on counties excluded entirely from model training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

FEATURES = [f"f{i}" for i in range(8)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    x = data[FEATURES]
    y = data["label"]
    folds = []
    aucs = []

    splitter = LeaveOneGroupOut()
    for train_idx, val_idx in splitter.split(x, y, groups=data["region"]):
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        model.fit(train[FEATURES], train["label"])
        scores = model.predict_proba(val[FEATURES])[:, 1]
        aucs.append(roc_auc_score(val["label"], scores))
        folds.append(
            {
                "train_ids": train["row_id"].tolist(),
                "val_ids": val["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "The starter trains and validates only on Nyeri, where shade is a strong spurious "
            "disease cue, so its AUC is about 0.97. Across the other counties that shortcut "
            "does not transfer and deployment AUC falls to about 0.69. Leave-one-county-out "
            "validation tests whether the model generalizes to unseen regions."
        ),
    }
