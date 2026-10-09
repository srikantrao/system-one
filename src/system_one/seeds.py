"""Seed helpers for later execution stages; preparation does not run a model."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, torch: Any = None, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and an explicitly supplied PyTorch module.

    No heavyweight framework is imported by the foundation. Future adapters must
    call this before constructing models/loaders, and seed their loader workers.
    Determinism remains dependent on hardware, kernels, and framework versions.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
