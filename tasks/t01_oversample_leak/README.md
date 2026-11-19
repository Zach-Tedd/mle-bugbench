# t01 — Oversampling leak

**Difficulty:** medium · **Skills:** validation design, imbalanced classification, leakage detection · **Runtime:** < 10 s on CPU

## Instructions (shown to the solver)

A teammate trained a fraud classifier on `data.py`'s dataset and reports **0.99 accuracy**. They want to ship it. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported performance untrustworthy, and return an honest estimate of how the model will perform on new transactions.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `train_ids` | list[int] | `row_id`s used to fit the model (repeats allowed if you resample) |
| `test_ids` | list[int] | `row_id`s you evaluated on, each once |
| `test_scores` | list[float] | predicted fraud probability per `test_ids` entry, same order |
| `metric` | str | name of the metric you report |
| `value` | float | that metric on your test set |
| `notes` | str | what was wrong and what you changed |

Constraints: scikit-learn, pandas and NumPy only; must be deterministic for a given seed.

## Grade

```bash
uv run python -m bugbench.runner grade t01 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
