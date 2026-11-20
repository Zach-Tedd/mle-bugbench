"""Build the exact prompt given to an AI assistant for one task.

    uv run python -m bugbench.prompt t01 neutral | clip.exe   # copy to the Windows clipboard

Four conditions, from least to most guidance:
- realistic: a preview of the data (shape, types, first rows, summaries) and the model code,
             + "improve it". No generator source, no scenario, no claimed score.
- blind:   the data generator's source and the model code + "improve it". No scenario.
- neutral: the scenario (including the claimed score) + "improve it".
- review:  the scenario + the task's review instruction ("fix what makes it untrustworthy").
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import pandas as pd

from bugbench.core import load_module, task_dirs

CONDITIONS = ("realistic", "blind", "neutral", "review")

NEUTRAL_ASK = (
    "Please improve this model and give me the complete updated solution.py. "
    "Keep the same run() interface described below."
)
NEUTRAL_NOTES = "What you changed and why"
BLIND_ASK = (
    "Here is our data generator (data.py) and model code (solution.py). "
    "Please improve the model and give me the complete updated solution.py. "
    "Keep the same run() interface described below."
)
REALISTIC_ASK = (
    "Here is a preview of our dataset and our model code (solution.py). "
    "run() receives the full dataset as a pandas DataFrame. "
    "Please improve the model and give me the complete updated solution.py. "
    "Keep the same run() interface described below."
)
CLOSING = "Return the complete solution.py in a single Python code block."


def _instructions(task_dir: Path) -> str:
    text = (task_dir / "README.md").read_text(encoding="utf-8")
    start = text.index("## Instructions")
    start = text.index("\n", start) + 1
    end = text.index("## Grade")
    return text[start:end].strip()


def _sections(task_dir: Path) -> tuple[str, str, str]:
    """Split the task instructions into (context, review instruction, interface)."""
    paragraphs = [p.strip() for p in _instructions(task_dir).split("\n\n") if p.strip()]
    review_idx = next(i for i, p in enumerate(paragraphs) if p.startswith("Review it"))
    context = "\n\n".join(paragraphs[:review_idx])
    review = paragraphs[review_idx]
    interface = "\n\n".join(paragraphs[review_idx + 1 :])
    return context, review, interface


def _strip_module_docstring(code: str) -> str:
    """Drop a leading module docstring (it may describe the scenario or the claimed score)."""
    stripped = code.lstrip()
    if not stripped.startswith('"""'):
        return code
    end = stripped.index('"""', 3) + 3
    return stripped[end:].lstrip("\n")


def _code_block(name: str, path: Path, strip_docstring: bool = False) -> str:
    code = path.read_text(encoding="utf-8")
    if strip_docstring:
        code = _strip_module_docstring(code)
    return f"`{name}`:\n\n```python\n{code.rstrip()}\n```"


def data_preview(task_dir: Path) -> str:
    """What an engineer sees of the data: size, column types, first rows, summaries.
    Built from the exact dataset the grader uses (seed 0)."""
    data: pd.DataFrame = load_module(task_dir / "data.py", f"{task_dir.name}_preview").load_data(0)
    numeric = data.select_dtypes("number").columns
    other = [c for c in data.columns if c not in numeric]
    with pd.option_context(
        "display.max_columns",
        12,
        "display.max_rows",
        24,
        "display.width",
        120,
        "display.precision",
        3,
    ):
        parts = [
            f"Dataset preview: {len(data):,} rows x {data.shape[1]:,} columns.",
            "Column types:\n" + data.dtypes.astype(str).value_counts().to_string(),
            "First rows:\n" + data.head(8).to_string(max_cols=12),
            "Numeric summary:\n" + data[numeric].describe().T.to_string(max_rows=24),
        ]
        for col in other:
            counts = data[col].value_counts()
            if len(counts) <= 12:
                parts.append(f"Values of `{col}`:\n" + counts.to_string())
            else:
                parts.append(
                    f"`{col}`: {data[col].nunique():,} distinct values, "
                    f"from {data[col].min()} to {data[col].max()}."
                )
        if "label" in data.columns:
            parts.append("Label counts:\n" + data["label"].value_counts().to_string())
    return "```text\n" + "\n\n".join(parts) + "\n```"


def data_helpers(task_dir: Path) -> str | None:
    """Helper code from data.py that a solution needs (e.g. t05's candidate_subsets), without
    the data loader and without constants the helpers don't use. None if there is nothing."""
    source = (task_dir / "data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "load_data"]
    if not funcs:
        return None
    used = {n.id for f in funcs for n in ast.walk(f) if isinstance(n, ast.Name)}
    assigns = [n for n in tree.body if isinstance(n, ast.Assign | ast.AnnAssign)]
    kept: list[ast.stmt] = []
    changed = True
    while changed:
        changed = False
        for node in assigns:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if node not in kept and names & used:
                kept.append(node)
                used |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                changed = True
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    nodes: list[ast.stmt] = [*imports, *kept, *funcs]
    keep = sorted(nodes, key=lambda n: n.lineno)
    return "\n\n".join(ast.get_source_segment(source, n) or "" for n in keep)


def build_prompt(task_dir: Path, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    context, review, interface = _sections(task_dir)
    if condition != "review":
        # The task README describes `notes` as "What was wrong ..."; that would hint at a bug.
        interface = re.sub(
            "what was wrong and what you changed", NEUTRAL_NOTES, interface, flags=re.I
        )
    if condition == "realistic":
        starter = _code_block(
            "solution.py", task_dir / "starter" / "solution.py", strip_docstring=True
        )
        helpers = data_helpers(task_dir)
        helper_block = (
            [
                f"`data.py` (helper functions only; the data loader is not shown):\n\n"
                f"```python\n{helpers}\n```"
            ]
            if helpers
            else []
        )
        parts = [REALISTIC_ASK, interface, data_preview(task_dir), *helper_block, starter, CLOSING]
        return "\n\n".join(parts) + "\n"
    blind = condition == "blind"
    code = [
        _code_block("data.py", task_dir / "data.py", strip_docstring=blind),
        _code_block("solution.py", task_dir / "starter" / "solution.py", strip_docstring=blind),
    ]
    if blind:
        parts = [BLIND_ASK, interface, *code, CLOSING]
    else:
        ask = NEUTRAL_ASK if condition == "neutral" else review
        parts = [
            context.replace("starter/solution.py", "solution.py"),
            ask,
            interface,
            *code,
            CLOSING,
        ]
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bugbench.prompt")
    parser.add_argument("task", help="task folder prefix, e.g. t01")
    parser.add_argument("condition", choices=CONDITIONS)
    args = parser.parse_args(argv)
    matches = [d for d in task_dirs() if d.name.startswith(args.task)]
    if len(matches) != 1:
        print(f"no unique task matching {args.task!r}", file=sys.stderr)
        return 2
    sys.stdout.write(build_prompt(matches[0], args.condition))
    return 0


if __name__ == "__main__":
    sys.exit(main())
