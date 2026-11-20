from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["t", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]


def _build_features(work: pd.DataFrame) -> pd.DataFrame:
    # Only calendar-derived features: the day index and day of week are known
    # in advance, so nothing here uses information from the future.
    start = work["date"].min()
    work["t"] = (work["date"] - start).dt.days.astype(float)
    dow = work["date"].dt.day_of_week
    for d in range(7):
        work[f"dow_{d}"] = (dow == d).astype(float)
    return work


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = data.sort_values("date").reset_index(drop=True).copy()
    work = _build_features(work)

    # Forward-chaining (expanding window) validation: every validation fold
    # lies strictly after its training data, so there is no temporal leakage.
    splitter = TimeSeriesSplit(n_splits=5)
    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(work):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]
        model = LinearRegression()  # deterministic closed-form fit
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
        "Fixed two leakage problems in the original: (1) roll7 was a centered "
        "rolling mean, which included the current day's demand and the next 3 "
        "days (target/future leakage); (2) shuffled KFold let the model train on "
        "days after the ones it was validated on. Replaced with forward-chaining "
        "TimeSeriesSplit and removed all demand-derived features. The data is a "
        "linear trend + fixed weekly pattern + iid noise, so lags and rolling "
        "means add nothing beyond noise; a linear model on a time index plus "
        "day-of-week dummies matches the true structure, extrapolates the trend "
        "(which trees cannot), and is deterministic. The reported MAE is a "
        "realistic out-of-time estimate, not comparable to the earlier "
        "leaky number."
    )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES,
        "notes": notes,
    }