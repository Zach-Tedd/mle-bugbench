from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(10)]


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: mean of each feature over the patient's visits,
    plus visit count. Uses no labels, so it is safe to apply to test patients."""
    g = df.groupby("patient_id")
    table = g[FEATURES].mean()
    table["n_visits"] = g.size()
    return table


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 1. Split by PATIENT (not by visit) to avoid leakage. Each patient has a
    #    large persistent offset that lets a model memorize identities.
    patients = data.groupby("patient_id")["label"].first()
    train_p, test_p = train_test_split(
        patients.index.to_numpy(),
        test_size=0.2,
        stratify=patients.to_numpy(),
        random_state=seed,
    )
    train = data[data["patient_id"].isin(train_p)]
    test = data[data["patient_id"].isin(test_p)]

    # 2. Patient-level features: averaging visits shrinks visit noise (sd 1)
    #    by sqrt(n_visits); the persistent offset is handled by the model.
    x_train = _patient_table(train)
    x_test = _patient_table(test)
    y_train = patients.loc[x_train.index].to_numpy()

    # Use the 10 averaged features (n_visits is only used for weighting below).
    cols = FEATURES

    # 3. Regularized logistic regression. The true signal is linear (a shift
    #    on f0-f2), so a linear model is better matched than a forest, which
    #    chases the nuisance offsets in f3-f9 and is data-hungry with only
    #    ~320 training patients.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=1000, random_state=seed),
    )
    model.fit(x_train[cols], y_train)

    # 4. Score each test patient once, then assign that score to each of the
    #    patient's visits (rows) so test_ids/test_scores cover every test visit.
    patient_scores = pd.Series(
        model.predict_proba(x_test[cols])[:, 1], index=x_test.index
    )
    scores = test["patient_id"].map(patient_scores).to_numpy(dtype=float)

    # 5. Metric: visit-level ROC AUC on the held-out patients' visits, which
    #    matches the original visit-level evaluation.
    value = float(roc_auc_score(test["label"], scores))

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": value,
        "notes": (
            "Replaced the visit-level random split with a stratified "
            "patient-level split: visits from one patient were appearing in "
            "both train and test, and the large per-patient offsets let the "
            "random forest memorize patients, inflating AUC. Now fits and "
            "evaluates on disjoint patients, so the reported AUC is an honest "
            "estimate of generalization to new patients. Modeling: aggregate "
            "each patient's visits to per-patient feature means (cuts visit "
            "noise by ~sqrt(n_visits)) and fit a standardized, L2-regularized "
            "logistic regression, which matches the linear disease shift on "
            "f0-f2 better than a random forest on ~320 patients. Each test "
            "patient's score is assigned to all of their visits; AUC is "
            "computed over those visits. Deterministic given seed."
        ),
    }