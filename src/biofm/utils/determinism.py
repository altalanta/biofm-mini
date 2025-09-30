"""Deterministic training utilities for reproducible results."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds across all libraries and configure deterministic behavior.
    
    This is the main entry point for ensuring reproducible results.
    Sets seeds for Python, NumPy, PyTorch, and configures deterministic
    algorithms and DataLoader workers.
    
    Args:
        seed: Random seed to use across all libraries
    """
    # Set basic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    
    # Enable deterministic algorithms (warn if not supported)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass


def deterministic_worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for deterministic DataLoader behavior.
    
    This function should be passed to the DataLoader's worker_init_fn parameter
    to ensure that each worker process has a unique but deterministic seed.
    
    Args:
        worker_id: Worker ID provided by PyTorch DataLoader
    """
    # Get the current process seed and make it deterministic per worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_deterministic_dataloader_kwargs() -> dict[str, Any]:
    """Get DataLoader kwargs for deterministic behavior.
    
    Returns:
        Dictionary of kwargs to pass to DataLoader constructor for deterministic behavior
    """
    return {
        "worker_init_fn": deterministic_worker_init_fn,
        "generator": torch.Generator().manual_seed(42),
    }


__all__ = ["set_all_seeds", "deterministic_worker_init_fn", "get_deterministic_dataloader_kwargs"]