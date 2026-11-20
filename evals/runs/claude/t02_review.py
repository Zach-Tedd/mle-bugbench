"""Demand forecaster with leak-free features and forward-chaining validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["lag_1", "lag_2", "lag_7", "roll7", "roll28", "dow"]
N_FOLDS = 5
MIN_TRAIN = 365


def _build(data: pd.DataFrame) -> pd.DataFrame:
    work = data.sort_values("date").drop_duplicates("date").copy()
    work = work.set_index("date")
    full_idx = pd.date_range(work.index.min(), work.index.max(), freq="D")
    work = work.reindex(full_idx)
    work.index.name = "date"
    s = work["demand"]
    # All features use only demand up to the end of the previous day.
    work["lag_1"] = s.shift(1)
    work["lag_2"] = s.shift(2)
    work["lag_7"] = s.shift(7)
    work["roll7"] = s.shift(1).rolling(7).mean()
    work["roll28"] = s.shift(1).rolling(28).mean()
    work["dow"] = work.index.day_of_week
    work = work.reset_index()
    work = work.dropna(subset=FEATURES + ["demand", "row_id"]).reset_index(drop=True)
    work["row_id"] = work["row_id"].astype(int)
    return work


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = _build(data)
    n = len(work)

    # Forward-chaining: expanding training window, validation block strictly after it.
    val_size = (n - MIN_TRAIN) // N_FOLDS
    folds = []
    fold_maes = []

    for k in range(N_FOLDS):
        train_end = MIN_TRAIN + k * val_size
        val_end = n if k == N_FOLDS - 1 else train_end + val_size
        train = work.iloc[:train_end]
        val = work.iloc[train_end:val_end]

        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=8,
            min_child_samples=20,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(train[FEATURES], train["demand"])
        val_preds = model.predict(val[FEATURES])
        fold_maes.append(mean_absolute_error(val["demand"], val_preds))
        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": val_preds.astype(float).tolist(),
            }
        )

    notes = (
        "Problems found: (1) roll7 used a centered window, so it included the "
        "target day and future days (direct leakage) -> replaced with trailing "
        "means computed on demand shifted by one day (roll7, roll28), plus lag_1, "
        "lag_2, lag_7; all use only data through the previous day. "
        "(2) Shuffled KFold trained on future days and validated on past days and "
        "put adjacent, highly autocorrelated days on both sides of the split -> "
        "replaced with expanding-window forward-chaining validation where every "
        "validation block lies strictly after its training data. "
        "(3) Data is reindexed to a complete daily calendar so shifts mean exactly "
        "'previous day'. (4) Model made a bit more regularized; trailing level "
        "features let the tree follow the upward trend. The reported MAE is the "
        "mean over time-ordered folds and is an honest next-day estimate; the "
        "irreducible noise (sd 5) puts a floor near MAE ~4."
    )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": list(FEATURES),
        "notes": notes,
    }