from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = [f"f{i}" for i in range(8)]


def _center_by_region(data: pd.DataFrame) -> pd.DataFrame:
    """Remove per-region feature offsets (label-free batch correction)."""
    x = data[FEATURES].copy()
    grp = data["region"]
    x = x - x.groupby(grp).transform("mean")
    x = x / x.groupby(grp).transform("std").replace(0, 1.0)
    return x


def _consistent_features(x: pd.DataFrame, y: pd.Series, regions: pd.Series,
                         min_effect: float = 0.3) -> list[str]:
    """Keep features whose class effect is present in EVERY training region.

    Features that separate classes in only one region (a local shortcut) are
    dropped because they will not transfer to unseen regions.
    """
    effects = []
    for r in regions.unique():
        m = (regions == r).values
        xr, yr = x[m], y[m]
        diff = xr[yr == 1].mean() - xr[yr == 0].mean()
        effects.append(diff.values)
    worst = np.min(np.vstack(effects), axis=0)
    keep = [f for f, e in zip(FEATURES, worst) if e > min_effect]
    return keep if keep else list(FEATURES)


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)
    x_all = _center_by_region(data)
    y_all = data["label"]
    regions = data["region"]

    folds = []
    aucs = []
    for held_out in sorted(regions.unique()):
        val_mask = (regions == held_out).values
        tr_mask = ~val_mask

        keep = _consistent_features(
            x_all[tr_mask], y_all[tr_mask], regions[tr_mask]
        )

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        model.fit(x_all.loc[tr_mask, keep], y_all[tr_mask])
        scores = model.predict_proba(x_all.loc[val_mask, keep])[:, 1]

        aucs.append(float(roc_auc_score(y_all[val_mask], scores)))
        folds.append(
            {
                "train_ids": data.loc[tr_mask, "row_id"].tolist(),
                "val_ids": data.loc[val_mask, "row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Changes: (1) The original trained and validated only on Nyeri, where "
        "f3 carries a Nyeri-specific signal, so its AUC was optimistic and would "
        "not transfer to other regions. Now evaluating with leave-one-region-out "
        "folds across all 5 regions (1200 train / 300 val per fold), which "
        "measures generalization to unseen regions. The reported mean is "
        "therefore not directly comparable to the old Nyeri-only number and is "
        "expected to be lower but honest. (2) Per-region centering/scaling of "
        "features (uses features only, no labels; it is computed on each region's "
        "own features, including the held-out region) removes region offsets. "
        "(3) Within each training set, drop features whose class effect is not "
        "consistent across training regions (removes the Nyeri-only f3 shortcut). "
        "(4) Logistic regression instead of random forest: the signal is linear "
        "and Gaussian, so a low-variance linear model fits better with 1200 rows. "
        "Deterministic; no randomness beyond the seed passed to the solver."
    )
    return {
        "folds": folds,
        "metric": "roc_auc_leave_one_region_out",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }