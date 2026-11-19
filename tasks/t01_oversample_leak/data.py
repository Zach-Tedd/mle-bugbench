"""Deterministic fraud-style dataset with realistic exact-duplicate rows.

About 3% of rows are fraudulent. Fraud attempts are often retried, so
positive rows are much more likely to appear more than once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURES = [f"f{i}" for i in range(12)]


def load_data(seed: int = 0) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=6000,
        n_features=len(FEATURES),
        n_informative=6,
        n_redundant=2,
        weights=[0.97],
        flip_y=0.01,
        class_sep=0.8,
        random_state=seed,
    )
    frame = pd.DataFrame(x.round(6), columns=FEATURES)
    frame["label"] = y

    rng = np.random.default_rng(seed)
    p = np.where(frame["label"].to_numpy() == 1, 20.0, 1.0)
    dup_idx = rng.choice(len(frame), size=300, replace=True, p=p / p.sum())
    frame = pd.concat([frame, frame.iloc[dup_idx]], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame
