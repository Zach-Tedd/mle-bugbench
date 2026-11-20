
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


def _make_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build strictly causal features.

    Features for day t use only observations available through day t-1.
    """
    work = data.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["date", "row_id"]).reset_index(drop=True)

    dates = work["date"]

    # Calendar features.
    dow = dates.dt.dayofweek
    doy = dates.dt.dayofyear
    month = dates.dt.month
    week = dates.dt.isocalendar().week.astype(int)

    work["dow"] = dow
    work["day_of_month"] = dates.dt.day
    work["month"] = month
    work["week_of_year"] = week
    work["is_weekend"] = (dow >= 5).astype(int)

    # Cyclical calendar encodings.
    work["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    work["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    work["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    work["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    work["month_sin"] = np.sin(2 * np.pi * month / 12)
    work["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Historical demand features.
    # Shift first so no feature contains current-day demand.
    shifted = work["demand"].shift(1)

    lag_periods = [1, 2, 3, 6, 7, 8, 14, 21, 28, 35, 56]

    for lag in lag_periods:
        work[f"lag_{lag}"] = work["demand"].shift(lag)

    # Rolling statistics over past observations only.
    for window in [3, 7, 14, 28, 56]:
        work[f"roll_mean_{window}"] = (
            shifted.rolling(window, min_periods=window).mean()
        )
        work[f"roll_std_{window}"] = (
            shifted.rolling(window, min_periods=window).std()
        )

    # Recent demand trend.
    work["trend_7_28"] = (
        work["roll_mean_7"] - work["roll_mean_28"]
    )

    work["trend_7_14"] = (
        work["roll_mean_7"] - work["roll_mean_14"]
    )

    # Same weekday historical demand.
    work["same_dow_mean_4"] = (
        work["demand"]
        .shift(1)
        .rolling(28, min_periods=28)
        .mean()
    )

    features = [
        "dow",
        "day_of_month",
        "month",
        "week_of_year",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "month_sin",
        "month_cos",
    ]

    features += [f"lag_{lag}" for lag in lag_periods]

    for window in [3, 7, 14, 28, 56]:
        features += [
            f"roll_mean_{window}",
            f"roll_std_{window}",
        ]

    features += [
        "trend_7_28",
        "trend_7_14",
        "same_dow_mean_4",
    ]

    work = work.dropna(subset=features + ["demand"]).reset_index(drop=True)

    return work, features


def _make_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=250,
        learning_rate=0.025,
        num_leaves=7,
        max_depth=3,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Train and evaluate a leakage-free time-series regression model.

    Returns:
        folds: Validation predictions and corresponding row IDs.
        metric: Evaluation metric.
        value: Mean validation MAE.
        features: Model feature names.
        notes: Summary of improvements.
    """
    work, features = _make_features(data)

    # Expanding-window validation.
    # Every validation period is strictly after its training period.
    n_splits = 5
    n_samples = len(work)

    initial_train_size = n_samples // (n_splits + 1)

    folds = []
    fold_maes = []

    for fold in range(n_splits):
        train_end = initial_train_size * (fold + 1)
        val_end = (
            initial_train_size * (fold + 2)
            if fold < n_splits - 1
            else n_samples
        )

        train = work.iloc[:train_end].copy()
        val = work.iloc[train_end:val_end].copy()

        if len(train) == 0 or len(val) == 0:
            continue

        model = _make_model(seed + fold)

        model.fit(
            train[features],
            train["demand"],
        )

        val_preds = model.predict(val[features])

        fold_mae = mean_absolute_error(
            val["demand"],
            val_preds,
        )

        fold_maes.append(float(fold_mae))

        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": np.asarray(
                    val_preds,
                    dtype=float,
                ).tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": features,
        "notes": (
            "Replaced shuffled KFold with expanding-window temporal "
            "validation. Removed centered rolling features and added "
            "strictly causal lag and rolling statistics. Added calendar "
            "and cyclical seasonality features, historical demand "
            "trends, and regularized LightGBM hyperparameters. "
            "All validation predictions use only historical demand "
            "available before the prediction date. Training and "
            "evaluation are deterministic for a fixed seed."
        ),
    }
