# t06 — Regional sampling bias

**Difficulty:** medium · **Skills:** external validation, distribution shift, leakage detection · **Runtime:** about 10 s on CPU · **Leakage type:** L3.3 sampling bias in test distribution

## Instructions (shown to the solver)

An agritech team built a crop-disease classifier from image features collected on farms in five Kenyan counties (`data.py`: Nyeri, Kiambu, Meru, Embu, Kirinyaga; 300 samples each). They trained and validated on Nyeri, their best-documented county, report **ROC-AUC 0.97**, and plan a national rollout. Their code is in `starter/solution.py`.

Review it, fix every problem that makes the reported performance untrustworthy, and return an honest estimate of how the model will perform in **counties it was not trained on**. Keep a model that genuinely learns.

Your `solution.py` must define `run(data: pd.DataFrame, seed: int = 0) -> dict` returning:

| Key | Type | Meaning |
|---|---|---|
| `folds` | list[dict] | One entry per validation fold, each with `train_ids`, `val_ids` (lists of `row_id`) and `val_scores` (predicted disease probability in `val_ids` order) |
| `metric` | str | Name of the metric you report |
| `value` | float | Mean of the per-fold metric |
| `notes` | str | What was wrong and what you changed |

Constraints: scikit-learn, pandas and NumPy only; must be deterministic for a given seed.

## Grade

```bash
uv run python -m bugbench.runner grade t06 path/to/solution.py
```

See [rubric.md](rubric.md) for the criteria and weights.
