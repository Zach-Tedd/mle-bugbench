"""Deterministic seeding for Python, NumPy and (if installed) PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed every RNG this project may touch.

    PyTorch is optional: it is seeded only when installed, so CPU-only
    scikit-learn projects don't need it as a dependency.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker) so each worker is reproducible."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
