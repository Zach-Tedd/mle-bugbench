# t04 — Feature selection before cross-validation

**Difficulty:** medium · **Skills:** pipeline design, high-dimensional data, leakage detection · **Runtime:** about 10 s on CPU · **Leakage type:** L1.3 feature selection on training and test set

## Instructions (shown to the solver)

A lab is searching for biomarkers of drug response (`data.py`: 200 tumour samples, 2,000 gene-expression features). An analyst selected the 50 genes most associated with response, trained a logistic regression, and reports a cross-validated **ROC-AUC of 0.94**. They want to publish. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported performance untrustworthy, and return an honest cross-validated estimate for **new patients**. Keep a model that genuinely learns from the data.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `folds` | list[dict] | One entry per fold, each with `train_ids`, `val_ids` (lists of `row_id`) and `val_scores` (predicted probabilities in `val_ids` order) |
| `metric` | str | Name of the metric you report |
| `value` | float | Mean of the per-fold metric |
| `notes` | str | What was wrong and what you changed |

Constraints: scikit-learn, pandas and NumPy only; must be deterministic for a given seed; every sample must be validated exactly once.

## Grade

```bash
uv run python -m bugbench.runner grade t04 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
