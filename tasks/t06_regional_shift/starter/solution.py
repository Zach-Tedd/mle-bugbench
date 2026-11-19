"""Crop-disease model. The team validated on Nyeri farms (ROC-AUC 0.97)
and plans a national rollout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(8)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    nyeri = data[data["region"] == "Nyeri"]
    train, val = train_test_split(
        nyeri,
        test_size=0.25,
        stratify=nyeri["label"],
        random_state=seed,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(val[FEATURES])[:, 1]

    auc = float(roc_auc_score(val["label"], scores))
    return {
        "folds": [
            {
                "train_ids": train["row_id"].tolist(),
                "val_ids": val["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        ],
        "metric": "roc_auc",
        "value": float(np.mean([auc])),
        "notes": "",
    }
