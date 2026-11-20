import pytest

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


@pytest.mark.parametrize(
    "name",
    [
        "patient_level_roc_auc",  # ChatGPT, t03
        "roc_auc_patient_level",  # Claude, t03: same meaning, other word order
        "roc_auc_leave_one_region_out",  # Claude, t06
        "roc_auc_leave_one_county_out",  # Claude, t06
        "mean ROC-AUC",
        "auroc_per_county",
    ],
)
def test_qualified_roc_auc_names_accepted_in_any_order(name: str) -> None:
    from bugbench.core import metric_is

    assert metric_is(name, "roc_auc")


@pytest.mark.parametrize("name", ["pr_auc", "patient_level_pr_auc", "accuracy", "auc_pr", "f1"])
def test_lookalike_names_are_not_roc_auc(name: str) -> None:
    from bugbench.core import metric_is

    assert not metric_is(name, "roc_auc")
