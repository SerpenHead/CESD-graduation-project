"""Reproducibility helpers: set all random seeds."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (CPU + CUDA).
    Must be called at the start of every script before model loading.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic ops where possible (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
