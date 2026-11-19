"""Task 06 tests guard regional generalization without requiring one exact protocol."""

from pathlib import Path

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t06_regional_shift"

POOLED_STRATIFIED_KFOLD = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
FEATURES = [f"f{i}" for i in range(8)]
def run(data, seed=0):
    folds, aucs = [], []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(data[FEATURES], data["label"]):
        train, val = data.iloc[train_idx], data.iloc[val_idx]
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        model.fit(train[FEATURES], train["label"])
        scores = model.predict_proba(val[FEATURES])[:, 1]
        aucs.append(roc_auc_score(val["label"], scores))
        folds.append({
            "train_ids": train["row_id"].tolist(),
            "val_ids": val["row_id"].tolist(),
            "val_scores": scores.tolist(),
        })
    return {"folds": folds, "metric": "roc_auc", "value": float(np.mean(aucs)), "notes": ""}
"""

NYERI_TO_OTHER_COUNTIES = """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
FEATURES = [f"f{i}" for i in range(8)]
def run(data: pd.DataFrame, seed=0):
    train = data[data["region"] == "Nyeri"]
    val = data[data["region"] != "Nyeri"]
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(val[FEATURES])[:, 1]
    auc = roc_auc_score(val["label"], scores)
    return {"folds": [{
        "train_ids": train["row_id"].tolist(),
        "val_ids": val["row_id"].tolist(),
        "val_scores": scores.tolist(),
    }], "metric": "roc_auc", "value": float(auc), "notes": ""}
"""

BEST_TWO_LOCO_FOLDS = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
FEATURES = [f"f{i}" for i in range(8)]
def run(data, seed=0):
    ranked = []
    splitter = LeaveOneGroupOut()
    for train_idx, val_idx in splitter.split(data[FEATURES], data["label"], groups=data["region"]):
        train, val = data.iloc[train_idx], data.iloc[val_idx]
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        model.fit(train[FEATURES], train["label"])
        scores = model.predict_proba(val[FEATURES])[:, 1]
        auc = float(roc_auc_score(val["label"], scores))
        ranked.append((auc, {
            "train_ids": train["row_id"].tolist(),
            "val_ids": val["row_id"].tolist(),
            "val_scores": scores.tolist(),
        }))
    selected = sorted(ranked, key=lambda item: item[0], reverse=True)[:2]
    folds = [fold for _, fold in selected]
    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean([auc for auc, _ in selected])),
        "notes": "",
    }
"""


def _failed(report: Report) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_reference_scores_full_marks() -> None:
    report = grade(TASK, TASK / "reference" / "solution.py")
    assert report.passed and report.score == report.max_score == 100


def test_starter_fails_exactly_county_isolation_and_coverage() -> None:
    report = grade(TASK, TASK / "starter" / "solution.py")
    assert _failed(report) == {
        "unseen_validation_counties",
        "multiple_validation_counties",
    }


def test_pooled_stratified_folds_fail_only_county_isolation(tmp_path: Path) -> None:
    path = tmp_path / "pooled_kfold.py"
    path.write_text(POOLED_STRATIFIED_KFOLD)
    report = grade(TASK, path)
    assert _failed(report) == {"unseen_validation_counties"}


def test_nyeri_train_other_counties_test_is_fair(tmp_path: Path) -> None:
    path = tmp_path / "nyeri_to_other_counties.py"
    path.write_text(NYERI_TO_OTHER_COUNTIES)
    report = grade(TASK, path)
    assert report.passed and report.score == report.max_score == 100


def test_best_two_loco_folds_fail_only_county_coverage(tmp_path: Path) -> None:
    path = tmp_path / "best_two_loco.py"
    path.write_text(BEST_TWO_LOCO_FOLDS)
    report = grade(TASK, path)
    assert _failed(report) == {"multiple_validation_counties"}


def test_empty_solution_scores_zero(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("def run(data, seed=0):\n    return {}\n")
    report = grade(TASK, path)
    assert not report.passed and report.score == 0
