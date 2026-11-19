# mle-bugbench

![ci](https://github.com/Zach-Tedd/mle-bugbench/actions/workflows/ci.yml/badge.svg)

A benchmark of realistic **machine-learning-engineering tasks with planted bugs**: data leakage, wrong validation, unstable training, slow pipelines, irreproducible results. Each task ships with solver instructions, buggy starter code, a reference solution, a weighted rubric, and an automatic grader.

Built to evaluate both people and AI coding models on the failures that actually break ML systems.

## Tasks

| Task | Planted bug | Skills tested | Reference | Starter |
|---|---|---|---|---|
| [t01_oversample_leak](tasks/t01_oversample_leak) | Oversampling before the split + duplicate rows; accuracy on a 3% base rate | Validation design, imbalance, leakage | 100/100 | 10/100 |

## How a task is built

```
tasks/tNN_name/
  README.md            instructions shown to the solver + interface
  rubric.md            weighted, objectively checkable criteria
  data.py              deterministic data (no downloads)
  starter/solution.py  the buggy code
  reference/solution.py  the fix
  grader.py            returns a scored Report
```

A grader is only trusted if it **passes the reference and fails the starter**, and the test suite also feeds it tempting partial fixes. CI enforces all of this on every push.

## Run it

```bash
make setup
make validate                                           # every grader vs reference and starter
uv run python -m bugbench.runner grade t01 my_solution.py  # grade your own attempt
make check                                              # lint, types, tests
```

## Roadmap

Tasks in progress: time-series CV leak (LightGBM), group leakage, NaN loss, nondeterministic DataLoader, data-loader bottleneck, broken freeze/unfreeze schedule. Then: model evaluations with failure analysis, and an LLM-as-judge for the manual criteria.
