"""Crop-disease features from 5 Kenyan counties; Nyeri photos carry a spurious shade cue."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(8)]
REGIONS = ["Nyeri", "Kiambu", "Meru", "Embu", "Kirinyaga"]
SAMPLES_PER_REGION = 300


def load_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for region in REGIONS:
        y = rng.binomial(1, 0.3, size=SAMPLES_PER_REGION)
        x = rng.normal(0.0, 1.0, size=(SAMPLES_PER_REGION, len(FEATURES)))
        x += rng.normal(0.0, 1.0, size=len(FEATURES))
        x[:, :3] += 0.8 * y[:, None]
        if region == "Nyeri":
            x[:, 3] += 2.5 * y

        frame = pd.DataFrame(x, columns=FEATURES)
        frame["region"] = region
        frame["label"] = y
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data.insert(0, "row_id", np.arange(len(data)))
    return data
