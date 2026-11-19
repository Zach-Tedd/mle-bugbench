"""Reference fix: keep all visits from each patient in one validation partition."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, data["label"], groups=data["patient_id"]))
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": float(roc_auc_score(test["label"], scores)),
        "notes": (
            "The row-wise split allowed visits from the same patient into training and test, "
            "so the forest could recognize patient fingerprints. StratifiedGroupKFold keeps "
            "each patient in one partition. A single held-out fold contains only about 80 "
            "patients, so its AUC has substantial sampling variance; group-aware cross-validation "
            "would give a more stable estimate."
        ),
    }
