"""Diagnosis model with leakage-free, patient-level evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(10)]


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: mean of each feature across visits."""
    g = df.groupby("patient_id", sort=True)
    agg = g[FEATURES].mean()
    agg["label"] = g["label"].first()
    return agg


def _make_model(C: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, random_state=seed),
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # Split by patient so no patient appears in both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(data, data["label"], groups=data["patient_id"])
    )
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    train_p = _patient_table(train)
    test_p = _patient_table(test)

    X_tr = train_p[FEATURES].to_numpy()
    y_tr = train_p["label"].to_numpy()

    # Choose regularization by grouped CV on training patients only.
    best_C, best_auc = 1.0, -np.inf
    cv = GroupKFold(n_splits=5)
    groups = np.arange(len(train_p))
    for C in (0.01, 0.1, 1.0, 10.0):
        aucs = []
        for a, b in cv.split(X_tr, y_tr, groups):
            if len(np.unique(y_tr[b])) < 2:
                continue
            m = _make_model(C, seed).fit(X_tr[a], y_tr[a])
            aucs.append(roc_auc_score(y_tr[b], m.predict_proba(X_tr[b])[:, 1]))
        mean_auc = float(np.mean(aucs)) if aucs else -np.inf
        if mean_auc > best_auc:
            best_C, best_auc = C, mean_auc

    model = _make_model(best_C, seed).fit(X_tr, y_tr)

    # Patient-level scores, mapped back to each test visit.
    test_p_scores = pd.Series(
        model.predict_proba(test_p[FEATURES].to_numpy())[:, 1],
        index=test_p.index,
    )
    visit_scores = test["patient_id"].map(test_p_scores).to_numpy()

    # Metric at patient level: one entry per patient.
    value = float(roc_auc_score(test_p["label"], test_p_scores.to_numpy()))

    notes = (
        "Original 0.93 AUC was leakage: visit-level random split put the same "
        "patients in train and test, and large per-patient offsets (SD 3) let the "
        "forest identify patients rather than disease. Fixes: (1) group split by "
        "patient; (2) aggregate visits to per-patient means to reduce visit noise "
        "(the disease signal is a 1.5 mean shift in f0-f2); (3) regularized "
        "logistic regression, C chosen by grouped CV on training patients only "
        f"(C={best_C}); (4) every test visit gets its patient's score, and ROC-AUC "
        "is computed per patient to avoid overweighting patients with many visits. "
        "Expect a much lower, honest AUC than 0.93, because between-patient offsets "
        "swamp the signal."
    )

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": [float(s) for s in visit_scores],
        "metric": "roc_auc_patient_level",
        "value": value,
        "notes": notes,
    }