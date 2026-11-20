
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # Reset positional indices while preserving original row IDs.
    data = data.reset_index(drop=True)

    X = data[FEATURES].astype(float)
    y = data["label"].astype(int)
    groups = data["patient_id"]

    # Group-aware, stratified split:
    # all visits from a patient stay in the same partition.
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Model 1: Extra Trees.
    extra_trees = ExtraTreesClassifier(
        n_estimators=600,
        max_features=0.85,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )

    # Model 2: Histogram gradient boosting.
    hist_gradient = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=seed,
    )

    # Model 3: RBF Support Vector Machine.
    svm = make_pipeline(
        StandardScaler(),
        SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
    )

    # Model 4: Regularized logistic regression.
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )

    # Train complementary models.
    extra_trees.fit(X_train, y_train)
    hist_gradient.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    logistic.fit(X_train, y_train)

    # Obtain disease probabilities.
    p_extra = extra_trees.predict_proba(X_test)[:, 1]
    p_hist = hist_gradient.predict_proba(X_test)[:, 1]
    p_svm = svm.predict_proba(X_test)[:, 1]
    p_logistic = logistic.predict_proba(X_test)[:, 1]

    # Weighted probability ensemble.
    scores = (
        0.30 * p_extra
        + 0.25 * p_hist
        + 0.30 * p_svm
        + 0.15 * p_logistic
    )

    # Evaluate on entirely held-out patients.
    value = roc_auc_score(y_test, scores)

    return {
        "train_ids": data.iloc[train_idx]["row_id"].astype(int).tolist(),
        "test_ids": data.iloc[test_idx]["row_id"].astype(int).tolist(),
        "test_scores": scores.astype(float).tolist(),
        "metric": "roc_auc",
        "value": float(value),
        "notes": (
            "Replaced random row-level splitting with "
            "StratifiedGroupKFold grouped by patient_id to prevent "
            "patient-level leakage. Excluded patient_id from features. "
            "Introduced a weighted probability ensemble of Extra Trees, "
            "Histogram Gradient Boosting, RBF-SVM, and regularized "
            "Logistic Regression. Standardized features for the "
            "distance-based models. All estimators use deterministic "
            "random states. Evaluation is performed on held-out patients."
        ),
    }
