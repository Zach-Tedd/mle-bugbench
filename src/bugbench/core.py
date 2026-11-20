"""Shared types and loaders used by every task's grader and by the runner."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"

# Names for each metric. "token" names may appear anywhere in a reported label as whole
# words ("roc_auc_leave_one_county_out", "patient_level_roc_auc"); "exact" names are too
# short or ambiguous to search for ("auc" also ends "pr_auc"; "ap" is a common word part).
# A grader must not fail a correct solution for naming or qualifying its metric differently.
METRIC_NAMES: dict[str, dict[str, frozenset[str]]] = {
    "pr_auc": {
        "token": frozenset({"pr_auc", "average_precision", "auprc"}),
        "exact": frozenset({"ap", "average_precision_score"}),
    },
    "roc_auc": {
        "token": frozenset({"roc_auc", "auroc"}),
        "exact": frozenset({"auc", "roc_auc_score"}),
    },
    "mae": {
        "token": frozenset({"mae", "mean_absolute_error"}),
        "exact": frozenset(),
    },
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _contains(tokens: list[str], phrase: str) -> bool:
    words = phrase.split("_")
    return any(tokens[i : i + len(words)] == words for i in range(len(tokens) - len(words) + 1))


def metric_is(reported: object, canonical: str) -> bool:
    """True if `reported` names the same metric as `canonical`, with any qualifier around it.

    Matching is on whole words, so "roc_auc_patient_level" and "patient_level_roc_auc"
    both name ROC-AUC, while "pr_auc" does not.
    """
    if not isinstance(reported, str):
        return False
    tokens = _tokens(reported)
    names = METRIC_NAMES.get(canonical, {"token": frozenset({canonical}), "exact": frozenset()})
    if "_".join(tokens) in names["exact"] | names["token"]:
        return True
    return any(_contains(tokens, phrase) for phrase in names["token"])


@dataclass(frozen=True)
class Check:
    """One objectively checkable rubric criterion."""

    name: str
    weight: int
    passed: bool
    detail: str


@dataclass
class Report:
    task: str
    solution: str
    checks: list[Check] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(c.weight for c in self.checks if c.passed)

    @property
    def max_score(self) -> int:
        return sum(c.weight for c in self.checks)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "solution": self.solution,
            "score": self.score,
            "max_score": self.max_score,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "weight": c.weight, "passed": bool(c.passed), "detail": c.detail}
                for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = [f"{self.task} :: {self.solution} -> {self.score}/{self.max_score}"]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name} ({c.weight}) {c.detail}")
        return "\n".join(lines)


def load_module(path: Path, name: str) -> ModuleType:
    """Import a module from a file path (tasks live outside the package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_dirs() -> list[Path]:
    return sorted(p for p in TASKS_DIR.iterdir() if p.is_dir() and (p / "grader.py").exists())
