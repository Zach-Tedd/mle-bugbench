"""Leakage-safe daily demand forecasting with expanding-window validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


FEATURES = [
    "lag_1",
    "lag_7",
    "dow",
    "roll7",
    "day_index",
]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Evaluate a one-day-ahead demand forecasting model using chronological,
    expanding-window cross-validation.

    Every feature for a given day uses only information available at
    the end of the preceding day.
    """

    required = {"row_id", "date", "demand"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(data) < 50:
        raise ValueError("Insufficient data for reliable time-series validation.")

    work = data[["row_id", "date", "demand"]].copy()

    work["date"] = pd.to_datetime(work["date"])

    if work["date"].duplicated().any():
        raise ValueError("Duplicate dates are not supported.")

    if work["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique.")

    if work["demand"].isna().any():
        raise ValueError("Demand contains missing values.")

    work = work.sort_values("date").reset_index(drop=True)

    # Calendar features are known in advance.
    work["dow"] = work["date"].dt.dayofweek

    # A deterministic time index captures the underlying trend.
    work["day_index"] = (
        work["date"] - work["date"].min()
    ).dt.days

    # All demand-derived features are shifted to exclude the current day.
    work["lag_1"] = work["demand"].shift(1)
    work["lag_7"] = work["demand"].shift(7)

    # Rolling statistics must use historical observations only.
    work["roll7"] = (
        work["demand"]
        .shift(1)
        .rolling(window=7, min_periods=7)
        .mean()
    )

    # Remove rows without sufficient historical information.
    work = work.dropna(subset=FEATURES).reset_index(drop=True)

    if len(work) < 30:
        raise ValueError("Insufficient usable historical observations.")

    # Chronological expanding-window validation.
    # Every validation fold occurs strictly after its training fold.
    splitter = TimeSeriesSplit(n_splits=5)

    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(work[FEATURES]):

        train = work.iloc[train_idx]
        val = work.iloc[val_idx]

        model = LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbosity=-1,
        )

        model.fit(
            train[FEATURES],
            train["demand"],
        )

        val_preds = model.predict(val[FEATURES])

        fold_mae = mean_absolute_error(
            val["demand"],
            val_preds,
        )

        fold_maes.append(float(fold_mae))

        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": val_preds.astype(float).tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES.copy(),
        "notes": (
            "The original MAE of 4.0 was not trustworthy because the "
            "evaluation used shuffled KFold, allowing future observations "
            "to appear in training while predicting earlier observations. "
            "Additionally, roll7 used a centered rolling window, which "
            "included future demand values and caused direct target leakage. "
            "Both problems have been corrected. Validation now uses "
            "five chronological, expanding-window folds, ensuring that "
            "training observations always precede validation observations. "
            "The rolling mean is shifted by one day, and lag features "
            "use only previously observed demand. Added lag_7 to capture "
            "weekly autocorrelation and day_index to capture the gradual "
            "upward trend. Day-of-week is retained because weekday and "
            "weekend demand patterns are predictable in advance. "
            "The reported value is the unweighted mean of the five "
            "validation-fold MAEs. It is a more realistic estimate of "
            "one-day-ahead forecasting performance, although actual "
            "future performance may differ due to changing demand "
            "patterns or noise."
        ),
    }