
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"f{i}" for i in range(8)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Train and evaluate an ensemble of Logistic Regression and
    shrinkage LDA using stratified five-fold cross-validation.

    Returns:
        folds: Validation predictions and corresponding row IDs.
        metric: Evaluation metric name.
        value: Mean validation ROC-AUC.
        notes: Description of modeling choices.
    """

    # Evaluate on Nyeri, matching the original evaluation setup.
    nyeri = data.loc[data["region"] == "Nyeri"].copy()

    if nyeri.empty:
        raise ValueError("No Nyeri observations found.")

    if nyeri["label"].nunique() != 2:
        raise ValueError("Nyeri data must contain both classes.")

    if nyeri[FEATURES].isnull().any().any():
        raise ValueError("Features contain missing values.")

    X = nyeri[FEATURES].to_numpy(dtype=float)
    y = nyeri["label"].to_numpy(dtype=int)
    row_ids = nyeri["row_id"].to_numpy()

    # Stratified CV preserves the disease prevalence in each fold.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    fold_aucs = []

    for train_idx, val_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # Model 1: Regularized Logistic Regression.
        # Scaling is fitted independently inside each training fold.
        logistic = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ])

        logistic.fit(X_train, y_train)

        logistic_scores = logistic.predict_proba(X_val)[:, 1]

        # Model 2: Linear Discriminant Analysis with shrinkage.
        # Shrinkage stabilizes covariance estimates on small datasets.
        lda = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )

        lda.fit(X_train, y_train)

        lda_scores = lda.predict_proba(X_val)[:, 1]

        # Soft-voting ensemble.
        # Both models capture the underlying linear disease signal.
        scores = (
            0.7 * logistic_scores
            + 0.3 * lda_scores
        )

        auc = roc_auc_score(y_val, scores)
        fold_aucs.append(float(auc))

        folds.append({
            "train_ids": row_ids[train_idx].tolist(),
            "val_ids": row_ids[val_idx].tolist(),
            "val_scores": scores.tolist(),
        })

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(fold_aucs)),
        "notes": (
            "Replaced Random Forest with a 70/30 soft-voting ensemble "
            "of standardized regularized Logistic Regression and "
            "shrinkage LDA. The synthetic data has predominantly "
            "linear class-dependent feature shifts, including a "
            "strong Nyeri-specific signal in f3. Five-fold stratified "
            "cross-validation provides a more stable evaluation than "
            "a single random split. Each model is fitted independently "
            "within each training fold to prevent validation leakage. "
            "All randomness is controlled by the supplied seed."
        ),
    }
