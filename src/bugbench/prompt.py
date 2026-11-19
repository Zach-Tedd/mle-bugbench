"""Build the exact prompt given to an AI assistant for one task.

    uv run python -m bugbench.prompt t01 neutral | clip.exe   # copy to the Windows clipboard

Two conditions:
- neutral: context + "improve it". No hint that anything is wrong (how people really ask).
- review:  context + the task's review instruction ("fix what makes it untrustworthy").
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from bugbench.core import task_dirs

CONDITIONS = ("neutral", "review")

NEUTRAL_ASK = (
    "Please improve this model and give me the complete updated solution.py. "
    "Keep the same run() interface described below."
)
NEUTRAL_NOTES = "What you changed and why"
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


def _code_block(name: str, path: Path) -> str:
    return f"`{name}`:\n\n```python\n{path.read_text(encoding='utf-8').rstrip()}\n```"


def build_prompt(task_dir: Path, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    context, review, interface = _sections(task_dir)
    ask = NEUTRAL_ASK if condition == "neutral" else review
    if condition == "neutral":
        # The task README describes `notes` as "What was wrong ..."; that would hint at a bug.
        interface = re.sub(
            "what was wrong and what you changed", NEUTRAL_NOTES, interface, flags=re.I
        )
    parts = [
        context.replace("starter/solution.py", "solution.py"),
        ask,
        interface,
        _code_block("data.py", task_dir / "data.py"),
        _code_block("solution.py", task_dir / "starter" / "solution.py"),
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
