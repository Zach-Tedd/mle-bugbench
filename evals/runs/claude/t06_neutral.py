"""Crop-disease model with leave-one-county-out validation.

The original 0.97 ROC-AUC came from a Nyeri-only random split, where feature
f3 carries a county-specific (spurious) cue. This version validates on
held-out counties, drops the unstable feature, removes per-county offsets,
and uses a regularized linear model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALL_FEATURES = [f"f{i}" for i in range(8)]
# f3 is strongly predictive only in Nyeri (shade cue); not transferable.
FEATURES = [f for f in ALL_FEATURES if f != "f3"]


def _center_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-region feature means (label-free, so usable on a new county)."""
    out = df.copy()
    means = out.groupby("region")[FEATURES].transform("mean")
    out[FEATURES] = out[FEATURES] - means
    return out


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = _center_by_region(data)
    regions = sorted(data["region"].unique())

    folds = []
    aucs = []
    per_region = {}
    for held_out in regions:
        train = data[data["region"] != held_out]
        val = data[data["region"] == held_out]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000, random_state=seed),
        )
        model.fit(train[FEATURES], train["label"])
        scores = model.predict_proba(val[FEATURES])[:, 1]

        auc = float(roc_auc_score(val["label"], scores))
        aucs.append(auc)
        per_region[held_out] = round(auc, 3)
        folds.append(
            {
                "train_ids": train["row_id"].tolist(),
                "val_ids": val["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Original 0.97 AUC was misleading: it used a random split within Nyeri, "
        "where f3 carries a county-specific shade cue (+2.5*label) absent in other "
        "counties, so it would not transfer to a national rollout. Changes: "
        "(1) leave-one-county-out validation, which estimates performance on unseen "
        "counties; (2) dropped f3, the non-transferable feature; (3) per-county "
        "feature centering (label-free) to remove site offsets; (4) regularized "
        "logistic regression instead of a random forest, matching the small linear "
        "signal in f0-f2. Per-county held-out AUC: "
        f"{per_region}. The mean is a lower but honest estimate of rollout "
        "performance. Confirm the f3 finding with agronomists and collect labeled "
        "data from more counties before deployment."
    )
    return {
        "folds": folds,
        "metric": "roc_auc_leave_one_county_out",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }