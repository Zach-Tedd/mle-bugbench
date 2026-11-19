"""Gene expression: 200 tumour samples x 2000 genes; 10 genes weakly predict drug response."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_SAMPLES = 200
N_GENES = 2000
N_INFORMATIVE = 10
FEATURES = [f"g{i}" for i in range(N_GENES)]


def load_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, 0.5, size=N_SAMPLES)
    x = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_GENES))
    x[:, :N_INFORMATIVE] += 0.6 * label[:, None]

    frame = pd.DataFrame(x, columns=FEATURES)
    frame["label"] = label
    frame.insert(0, "row_id", np.arange(N_SAMPLES))
    return frame
