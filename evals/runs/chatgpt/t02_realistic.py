from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def _make_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construct strictly causal time-series features.

    Every demand-derived feature for day t uses observations
    available no later than day t-1.
    """
    df = data.copy()
    dates = pd.to_datetime(df["date"])

    y = df["demand"].astype(float)
    shifted = y.shift(1)

    features = pd.DataFrame(index=df.index)

    # Calendar and trend features.
    features["trend"] = (
        dates - dates.min()
    ).dt.total_seconds() / 86400.0

    day_of_week = dates.dt.dayofweek
    day_of_year = dates.dt.dayofyear

    features["dow"] = day_of_week
    features["month"] = dates.dt.month
    features["day_of_month"] = dates.dt.day
    features["quarter"] = dates.dt.quarter
    features["week_of_year"] = dates.dt.isocalendar().week.astype(float)

    # Weekly seasonality.
    features["dow_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    features["dow_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # Annual seasonality.
    annual_phase = 2 * np.pi * (
        day_of_year - 1
    ) / 365.25

    for harmonic in (1, 2, 3):
        features[f"year_sin_{harmonic}"] = np.sin(
            harmonic * annual_phase
        )
        features[f"year_cos_{harmonic}"] = np.cos(
            harmonic * annual_phase
        )

    # Demand lags.
    for lag in (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 21, 28, 35, 42, 56):
        features[f"lag_{lag}"] = y.shift(lag)

    # Rolling statistics, always shifted before calculation.
    for window in (3, 7, 14, 21, 28, 56):
        rolling = shifted.rolling(
            window=window,
            min_periods=window
        )

        features[f"roll_mean_{window}"] = rolling.mean()
        features[f"roll_std_{window}"] = rolling.std()
        features[f"roll_min_{window}"] = rolling.min()
        features[f"roll_max_{window}"] = rolling.max()

    # Exponentially weighted moving averages.
    for span in (7, 14, 28, 56):
        features[f"ewm_{span}"] = shifted.ewm(
            span=span,
            adjust=False,
            min_periods=span
        ).mean()

    # Recent demand changes.
    features["diff_1_7"] = (
        features["lag_1"] - features["lag_7"]
    )

    features["diff_1_14"] = (
        features["lag_1"] - features["lag_14"]
    )

    features["diff_7_28"] = (
        features["lag_7"] - features["lag_28"]
    )

    features["diff_1_roll7"] = (
        features["lag_1"] - features["roll_mean_7"]
    )

    features["diff_roll7_roll28"] = (
        features["roll_mean_7"] - features["roll_mean_28"]
    )

    return features.replace(
        [np.inf, -np.inf], np.nan
    )


def _build_models(seed: int):
    """
    Create a regularized linear model and a nonlinear
    LightGBM model.
    """

    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=15.0)
    )

    lgbm = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=10,
        max_depth=4,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
    )

    return ridge, lgbm


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------

    work = data.copy()

    work["date"] = pd.to_datetime(work["date"])

    work = work.sort_values(
        ["date", "row_id"]
    ).reset_index(drop=True)

    if work["date"].duplicated().any():
        raise ValueError("Expected one observation per date.")

    if work["demand"].isna().any():
        raise ValueError("Demand contains missing values.")

    if len(work) < 200:
        raise ValueError("Insufficient observations for validation.")

    # ---------------------------------------------------------
    # 2. Generate causal features
    # ---------------------------------------------------------

    features = _make_features(work)

    target = work["demand"].astype(float)

    valid = features.notna().all(axis=1)

    work = work.loc[valid].reset_index(drop=True)
    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)

    feature_names = features.columns.tolist()

    # ---------------------------------------------------------
    # 3. Expanding-window chronological validation
    # ---------------------------------------------------------

    n = len(work)

    n_splits = 5

    # Ensure the initial training window contains
    # approximately one year of observations.
    initial_train_size = max(
        180,
        int(n * 0.33)
    )

    if initial_train_size >= n - n_splits:
        initial_train_size = n - n_splits

    boundaries = np.linspace(
        initial_train_size,
        n,
        n_splits + 1,
        dtype=int
    )

    folds = []
    fold_maes = []

    # ---------------------------------------------------------
    # 4. Train and evaluate each fold
    # ---------------------------------------------------------

    for fold_idx in range(n_splits):

        val_start = boundaries[fold_idx]
        val_end = boundaries[fold_idx + 1]

        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)

        X_train = features.iloc[train_idx]
        y_train = target.iloc[train_idx]

        X_val = features.iloc[val_idx]
        y_val = target.iloc[val_idx]

        # Separate models.
        ridge, lgbm = _build_models(
            seed=seed + fold_idx
        )

        # Fit only on observations preceding validation.
        ridge.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        # Predictions.
        ridge_preds = ridge.predict(X_val)
        lgbm_preds = lgbm.predict(X_val)

        # Ensemble.
        val_preds = (
            0.55 * ridge_preds
            + 0.45 * lgbm_preds
        )

        # Demand is nonnegative.
        val_preds = np.maximum(
            val_preds,
            0.0
        )

        # Evaluate chronological validation.
        fold_mae = mean_absolute_error(
            y_val,
            val_preds
        )

        fold_maes.append(float(fold_mae))

        # Preserve original row IDs and validation order.
        folds.append(
            {
                "train_ids": (
                    work.iloc[train_idx]["row_id"]
                    .astype(int)
                    .tolist()
                ),

                "val_ids": (
                    work.iloc[val_idx]["row_id"]
                    .astype(int)
                    .tolist()
                ),

                "val_preds": (
                    val_preds.astype(float).tolist()
                ),
            }
        )

    # ---------------------------------------------------------
    # 5. Return results
    # ---------------------------------------------------------

    return {
        "folds": folds,

        "metric": "mae",

        "value": float(np.mean(fold_maes)),

        "features": feature_names,

        "notes": (
            "Replaced shuffled KFold with five expanding-window "
            "chronological validation folds. Removed centered "
            "rolling statistics to eliminate future information "
            "leakage. Added multiple historical demand lags, "
            "backward-looking rolling means and standard deviations, "
            "exponential moving averages, demand differences, "
            "calendar features, Fourier annual seasonality, weekly "
            "seasonality, and a continuous time trend. Combined "
            "regularized Ridge regression and a shallow, "
            "regularized LightGBM model using a 55/45 weighted "
            "ensemble. Validation predictions use causal features "
            "and models trained exclusively on observations "
            "preceding the validation period. All models use "
            "deterministic settings."
        ),
    }