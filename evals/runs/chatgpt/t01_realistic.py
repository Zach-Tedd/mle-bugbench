
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # --------------------------------------------------
    # 1. Prepare data
    # --------------------------------------------------

    df = data.copy()

    # Remove invalid labels if present.
    df = df[df["label"].isin([0, 1])].copy()

    X = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    y = df["label"].astype(int)

    ids = df["row_id"].astype(int)

    # --------------------------------------------------
    # 2. Stratified train/test split
    # --------------------------------------------------

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --------------------------------------------------
    # 3. Missing-value handling
    # --------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # --------------------------------------------------
    # 4. Class imbalance handling
    # --------------------------------------------------

    n_negative = int((y_train == 0).sum())
    n_positive = int((y_train == 1).sum())

    sample_weights = np.where(
        y_train.to_numpy() == 1,
        len(y_train) / (2.0 * n_positive),
        len(y_train) / (2.0 * n_negative),
    )

    # --------------------------------------------------
    # 5. Extra Trees
    # --------------------------------------------------

    extra_trees = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.85,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    extra_trees.fit(X_train, y_train)

    pred_extra = extra_trees.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 6. Random Forest
    # --------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed + 1,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    pred_rf = random_forest.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 7. Histogram Gradient Boosting
    # --------------------------------------------------

    hist_gradient = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=seed + 2,
    )

    hist_gradient.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    pred_hist = hist_gradient.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 8. Gradient Boosting
    # --------------------------------------------------

    gradient = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.85,
        max_features=None,
        random_state=seed + 3,
    )

    gradient.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    pred_gradient = gradient.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 9. Ensemble
    # --------------------------------------------------

    scores = (
        0.30 * pred_extra
        + 0.20 * pred_rf
        + 0.25 * pred_hist
        + 0.25 * pred_gradient
    )

    scores = np.clip(scores, 0.0, 1.0)

    # --------------------------------------------------
    # 10. Evaluation
    # --------------------------------------------------

    value = roc_auc_score(y_test, scores)

    return {
        "train_ids": ids.iloc[train_idx].tolist(),
        "test_ids": ids.iloc[test_idx].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": float(value),
        "notes": (
            "Improved fraud detection using a stratified split "
            "before training to prevent oversampling leakage. "
            "Replaced random oversampling with balanced class "
            "weights and sample weights. Added median imputation "
            "and an ensemble of ExtraTrees, RandomForest, "
            "HistGradientBoosting, and GradientBoosting. "
            "The ensemble combines complementary nonlinear "
            "decision boundaries. Evaluated using ROC-AUC "
            "on an untouched, naturally imbalanced test set. "
            "All random states are controlled by seed."
        ),
    }
