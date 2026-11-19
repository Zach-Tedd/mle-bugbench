# t02 — Time-series CV leak

**Difficulty:** medium · **Skills:** time-series validation, feature engineering, leakage detection · **Runtime:** about 15 s on CPU · **Leakage types:** L3.1 temporal leakage, L2 illegitimate features

## Instructions (shown to the solver)

A retailer wants to forecast **tomorrow's demand** from past daily demand (`data.py`: three years of daily data with a trend, weekly peaks and noise). A teammate built a LightGBM model, reports a cross-validated **MAE of 4.0**, and wants to ship it. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported error untrustworthy, and return an honest estimate of how the model will perform on future days.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `folds` | list[dict] | One entry per validation fold, each with `train_ids`, `val_ids` (lists of `row_id`) and `val_preds` (predictions in `val_ids` order) |
| `metric` | str | Name of the metric you report |
| `value` | float | Mean of the per-fold metric |
| `features` | list[str] | Feature names the model used |
| `notes` | str | What was wrong and what you changed |

Constraints: pandas, NumPy, scikit-learn and LightGBM only; must be deterministic for a given seed. A prediction for a day may use only information available at the end of the previous day.

## Grade

```bash
uv run python -m bugbench.runner grade t02 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
