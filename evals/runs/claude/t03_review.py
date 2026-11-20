"""Diagnosis model with patient-level evaluation (no leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

FEATURES = [f"f{i}" for i in range(10)]


def _patient_split(data: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """Stratified split at the patient level: no patient in both sets."""
    rng = np.random.default_rng(seed)
    patients = data.groupby("patient_id")["label"].first()
    test_patients = []
    for lab in sorted(patients.unique()):
        ids = np.sort(patients.index[patients == lab].to_numpy())
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * test_frac)))
        test_patients.extend(ids[:k].tolist())
    test_mask = data["patient_id"].isin(set(test_patients))
    return data[~test_mask], data[test_mask]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    train, test = _patient_split(data, seed)
    assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    value = float(roc_auc_score(test["label"], scores))

    # Secondary diagnostic: patient-level AUC (mean visit score per patient).
    pat = (
        pd.DataFrame(
            {"p": test["patient_id"].to_numpy(), "s": scores, "y": test["label"].to_numpy()}
        )
        .groupby("p")
        .agg(s=("s", "mean"), y=("y", "first"))
    )
    patient_auc = float(roc_auc_score(pat["y"], pat["s"]))

    notes = (
        "Problem: the original split was random by visit, so the same patient "
        "appeared in train and test. Each patient has a large persistent feature "
        "offset (sd 3) that a random forest memorizes, letting it recall a known "
        "patient's label; the 0.93 AUC reflected patient re-identification, not "
        "disease detection on new patients. "
        "Changes: (1) split by patient_id, stratified on the patient label, so "
        "test patients are entirely unseen (asserted disjoint); (2) regularized the "
        "forest (min_samples_leaf=20) to limit memorization of patient-specific "
        "offsets; (3) each test visit is scored exactly once. "
        f"Reported value is visit-level ROC-AUC on held-out patients; "
        f"patient-level AUC (mean score per patient) was {patient_auc:.3f}. "
        f"Test set has {test['patient_id'].nunique()} patients; a single split "
        "is noisy, so expect a wide confidence interval (several AUC points). "
        "No hyperparameter tuning was done on the test set."
    )

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": [float(s) for s in scores],
        "metric": "roc_auc",
        "value": value,
        "notes": notes,
    }