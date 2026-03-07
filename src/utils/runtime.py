"""Runtime helpers for cross-platform path/device handling."""

from pathlib import Path
from typing import Any, Dict, Union

import torch


PathLike = Union[str, Path]


def get_inference_device() -> torch.device:
    """Return the preferred inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor-like values in a dict to target device."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def normalize_path(path: PathLike) -> Path:
    """Normalize user/config path to absolute Path."""
    return Path(path).expanduser().resolve()
