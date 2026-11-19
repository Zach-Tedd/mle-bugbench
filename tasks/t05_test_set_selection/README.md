# t05 — Model selection on the test set

**Difficulty:** medium · **Skills:** model selection, nested validation, leakage detection · **Runtime:** about 20 s on CPU · **Leakage type:** L1.1 no (real) test set

## Instructions (shown to the solver)

A maintenance team wants to predict machine failure from 30 sensor readings (`data.py`: 200 machines). An engineer tried **200 candidate models**, each using a different set of 5 sensors (`candidate_subsets()` in `data.py`), kept the one with the best test-set ROC-AUC, and reports that **ROC-AUC of 0.74**. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported performance untrustworthy, and return an honest estimate of how the chosen model will perform on **new machines**. You must still choose among the same 200 candidates, and keep a model that genuinely learns.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `train_ids` | list[int] | `row_id`s used for fitting and model selection |
| `test_ids` | list[int] | `row_id`s of the held-out test set, each once |
| `test_scores` | list[float] | Predicted failure probability per `test_ids` entry, same order |
| `metric` | str | Name of the metric you report |
| `value` | float | That metric on the test set |
| `notes` | str | What was wrong and what you changed |

Constraints: scikit-learn, pandas and NumPy only; must be deterministic for a given seed; hold out 25–35% of machines as the test set.

## Grade

```bash
uv run python -m bugbench.runner grade t05 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
