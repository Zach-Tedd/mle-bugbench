"""Machine-failure data: 200 machines x 30 sensor features; 3 sensors carry weak signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_SAMPLES = 200
N_SENSORS = 30
FEATURES = [f"s{i}" for i in range(N_SENSORS)]


def candidate_subsets() -> list[list[str]]:
    rng = np.random.default_rng(7)
    return [
        [FEATURES[i] for i in sorted(rng.choice(N_SENSORS, 5, replace=False))] for _ in range(200)
    ]


def load_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, 0.5, size=N_SAMPLES)
    x = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_SENSORS))
    x[:, :3] += 0.5 * label[:, None]

    frame = pd.DataFrame(x, columns=FEATURES)
    frame["label"] = label
    frame.insert(0, "row_id", np.arange(N_SAMPLES))
    return frame
