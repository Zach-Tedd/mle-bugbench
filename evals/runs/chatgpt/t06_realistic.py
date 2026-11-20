from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [f"f{i}" for i in range(8)]


def build_model(seed: int):
    """
    Build a diverse ensemble of tree-based and linear classifiers.

    Combining models with different inductive biases helps capture
    nonlinear feature interactions while retaining generalization.
    """

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.9,
        min_samples_leaf=2,
        min_samples_split=4,
        bootstrap=False,
        class_weight=None,
        random_state=seed,
        n_jobs=1,
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_features=0.8,
        min_samples_leaf=2,
        min_samples_split=4,
        max_depth=None,
        class_weight=None,
        random_state=seed + 1,
        n_jobs=1,
    )

    histogram_boosting = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=2.0,
        random_state=seed + 2,
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.85,
        max_features=0.9,
        random_state=seed + 3,
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight=None,
            random_state=seed + 4,
        ),
    )

    return [
        extra_trees,
        random_forest,
        histogram_boosting,
        gradient_boosting,
        logistic,
    ]


def predict_ensemble(
    models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """
    Fit individual models and average their predicted probabilities.
    """

    predictions = []

    for model in models:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        predictions.append(probabilities)

    # Equal-weight probability averaging reduces model-specific variance.
    scores = np.mean(np.vstack(predictions), axis=0)

    return np.clip(scores, 0.0, 1.0)


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # Preserve the original evaluation population.
    nyeri = data.loc[data["region"] == "Nyeri"].copy()

    if len(nyeri) == 0:
        raise ValueError("No observations found for region 'Nyeri'.")

    if nyeri["label"].nunique() != 2:
        raise ValueError("Nyeri data must contain both target classes.")

    # Keep only predictive numeric features.
    # row_id is an identifier and region is constant after filtering.
    X = nyeri[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = nyeri["label"].astype(int)

    # Median imputation protects against missing feature values.
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=FEATURES,
        index=nyeri.index,
    )

    # Five-fold stratified cross-validation provides a more stable
    # estimate than a single random train/validation split.
    n_splits = min(5, int(y.value_counts().min()))

    if n_splits < 2:
        raise ValueError("Insufficient observations for stratified CV.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Use independent deterministic seeds for each fold.
        fold_seed = seed + 100 * fold_idx

        models = build_model(fold_seed)

        scores = predict_ensemble(
            models,
            X_train,
            y_train,
            X_val,
        )

        auc = roc_auc_score(y_val, scores)
        fold_scores.append(float(auc))

        folds.append(
            {
                "train_ids": nyeri.iloc[train_idx]["row_id"].tolist(),
                "val_ids": nyeri.iloc[val_idx]["row_id"].tolist(),
                "val_scores": scores.astype(float).tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(fold_scores)),
        "notes": (
            "Replaced the single random holdout with five-fold stratified "
            "cross-validation to obtain a more stable ROC-AUC estimate. "
            "Introduced a probability-averaging ensemble consisting of "
            "ExtraTrees, RandomForest, HistGradientBoosting, "
            "GradientBoosting, and regularized LogisticRegression. "
            "The tree models capture nonlinear relationships and feature "
            "interactions, while logistic regression provides a complementary "
            "linear decision boundary. Added median imputation, excluded "
            "row_id to prevent identifier leakage, and used independent "
            "deterministic random seeds for reproducibility. "
            "All validation predictions are generated exclusively from "
            "models trained on their corresponding training folds."
        ),
    }