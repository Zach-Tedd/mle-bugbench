from bugbench.core import TASKS_DIR
from bugbench.runner import main


def test_grade_cli_passes_reference() -> None:
    ref = TASKS_DIR / "t01_oversample_leak" / "reference" / "solution.py"
    assert main(["grade", "t01", str(ref)]) == 0


def test_grade_cli_rejects_unknown_task() -> None:
    assert main(["grade", "t99", "x.py"]) == 2


def test_metric_aliases_accept_equivalent_names() -> None:
    from bugbench.core import metric_is

    assert metric_is("average_precision", "pr_auc")
    assert metric_is("AUROC", "roc_auc")
    assert metric_is("ROC-AUC", "roc_auc")
    assert metric_is("mean_absolute_error", "mae")
    assert not metric_is("accuracy", "pr_auc")
    assert not metric_is("roc_auc", "pr_auc")
    assert metric_is("patient_level_roc_auc", "roc_auc")
    assert not metric_is("patient_level_accuracy", "roc_auc")
    assert not metric_is(None, "mae")
