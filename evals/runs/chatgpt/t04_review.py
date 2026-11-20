
"""Leakage-safe cross-validated biomarker model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"g{i}" for i in range(2000)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Estimate performance on new patients using stratified 5-fold CV.

    All preprocessing, feature selection, and model fitting are
    performed exclusively on each fold's training samples.

    Returns:
        folds: Training IDs, validation IDs, and validation probabilities.
        metric: Evaluation metric name.
        value: Mean of the five fold-level ROC-AUC scores.
        notes: Explanation of corrections.
    """

    # Validate input.
    required_columns = FEATURES + ["label", "row_id"]
    missing = [c for c in required_columns if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique.")

    x = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy()
    row_ids = data["row_id"].tolist()

    if not np.isfinite(x).all():
        raise ValueError("Gene-expression features must be finite.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1).")

    if np.bincount(y.astype(int), minlength=2).min() < 5:
        raise ValueError("At least five samples per class are required.")

    # Stratified folds ensure both classes are represented.
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    aucs = []

    for train_idx, val_idx in splitter.split(x, y):

        # Every learned transformation is fitted only on training data.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(
                score_func=f_classif,
                k=50,
            )),
            ("classifier", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            )),
        ])

        # Fit without accessing validation labels.
        model.fit(x[train_idx], y[train_idx])

        # Predict probabilities for previously unseen patients.
        scores = model.predict_proba(x[val_idx])[:, 1]

        # Evaluate this fold independently.
        auc = roc_auc_score(y[val_idx], scores)
        aucs.append(auc)

        folds.append({
            "train_ids": [
                row_ids[i] for i in train_idx
            ],
            "val_ids": [
                row_ids[i] for i in val_idx
            ],
            "val_scores": scores.tolist(),
        })

    # Mean of per-fold ROC-AUC values.
    mean_auc = float(np.mean(aucs))

    notes = (
        "The original analysis suffered from feature-selection leakage: "
        "SelectKBest was fitted on the complete dataset before cross-validation, "
        "allowing validation labels to influence the selected genes and "
        "inflating the reported ROC-AUC. "
        "Feature selection is now performed independently within each "
        "training fold using a scikit-learn Pipeline. "
        "StandardScaler is also fitted independently within each training "
        "fold to prevent preprocessing leakage. "
        "L2-regularized logistic regression is retained as a genuine "
        "predictive model. "
        "Stratified 5-fold cross-validation ensures every sample is "
        "validated exactly once. "
        "The reported metric is the unweighted mean of the five "
        "validation-fold ROC-AUC scores. "
        "The resulting estimate is substantially more trustworthy "
        "for predicting responses in new patients, although its "
        "uncertainty and external generalizability should still be "
        "assessed before clinical or biomarker claims are made."
    )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": mean_auc,
        "notes": notes,
    }
