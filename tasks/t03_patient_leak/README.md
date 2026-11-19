# t03 — Patient leakage

**Difficulty:** medium · **Skills:** grouped validation, clinical ML, leakage detection · **Runtime:** about 5 s on CPU · **Leakage type:** L3.2 non-independence between train and test samples

## Instructions (shown to the solver)

A hospital team built a model that predicts a disease from routine clinical measurements (`data.py`: about 400 patients, 3–8 visits each, one diagnosis per patient). A teammate reports **ROC-AUC 0.93** and wants to publish. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported performance untrustworthy, and return an honest estimate of how the model will perform on **new patients**.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `train_ids` | list[int] | `row_id`s (visits) used to fit the model |
| `test_ids` | list[int] | `row_id`s you evaluated on, each once |
| `test_scores` | list[float] | Predicted disease probability per `test_ids` entry, same order |
| `metric` | str | Name of the metric you report |
| `value` | float | That metric on your test set |
| `notes` | str | What was wrong and what you changed |

Constraints: scikit-learn, pandas and NumPy only; must be deterministic for a given seed.

## Grade

```bash
uv run python -m bugbench.runner grade t03 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
