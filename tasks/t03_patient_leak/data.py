"""Clinical visits: about 400 patients, 3-8 visits each, one diagnosis per patient."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(10)]


def load_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_patients = 400
    patient_label = rng.binomial(1, 0.3, size=n_patients)
    patient_offset = rng.normal(0.0, 3.0, size=(n_patients, len(FEATURES)))
    rows = []

    for patient_id in range(n_patients):
        n_visits = int(rng.integers(3, 9))
        disease_signal = np.zeros(len(FEATURES))
        disease_signal[:3] = 1.5 * patient_label[patient_id]

        for _ in range(n_visits):
            features = (
                patient_offset[patient_id]
                + disease_signal
                + rng.normal(0.0, 1.0, size=len(FEATURES))
            )
            rows.append([patient_id, *features, int(patient_label[patient_id])])

    frame = pd.DataFrame(rows, columns=["patient_id", *FEATURES, "label"])
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame
