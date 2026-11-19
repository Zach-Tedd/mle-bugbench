"""Shared types and loaders used by every task's grader and by the runner."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


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
