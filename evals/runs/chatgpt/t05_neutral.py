
"""Machine failure prediction with leakage-free model selection.

Uses stratified cross-validation on the training set to select features
and logistic regression regularization. The held-out test set is used
only for final evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # Identify sensor columns.
    feature_cols = [
        col for col in data.columns
        if col not in ("row_id", "label")
    ]

    X = data[feature_cols]
    y = data["label"].astype(int)

    # Reserve a completely untouched 30% test set.
    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # All feature selection and preprocessing happen inside the
    # cross-validation pipeline to prevent information leakage.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selection", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )),
    ])

    # Tune the number of selected sensors and regularization strength.
    # The candidate feature counts are bounded by the available features.
    n_features = len(feature_cols)

    k_values = sorted({
        k for k in (3, 5, 10, 15, 20, 30)
        if k <= n_features
    })

    if not k_values:
        k_values = [n_features]

    param_grid = {
        "selection__k": k_values,
        "classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    }

    # Stratified CV ensures each fold retains a representative
    # proportion of failure and non-failure examples.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    # Model selection uses training data only.
    search.fit(X_train, y_train)

    # GridSearchCV refits the best pipeline on the complete training set.
    best_model = search.best_estimator_

    # Evaluate on the held-out test set exactly once.
    test_scores = best_model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, test_scores)

    return {
        "train_ids": [
            int(x) for x in data.iloc[train_idx]["row_id"].tolist()
        ],
        "test_ids": [
            int(x) for x in data.iloc[test_idx]["row_id"].tolist()
        ],
        "test_scores": [
            float(x) for x in test_scores
        ],
        "metric": "roc_auc",
        "value": float(test_auc),
        "notes": (
            "Replaced test-set-based selection among 200 candidate "
            "models with leakage-free stratified 5-fold cross-validation "
            "on the training set. Used a StandardScaler, SelectKBest "
            "with ANOVA F-statistics, and regularized LogisticRegression. "
            "Selected the number of sensors and regularization strength "
            "using training-only ROC-AUC. Refit the selected pipeline "
            "on all training data and evaluated once on the untouched "
            "30% test set. All randomness is controlled by seed."
        ),
    }
