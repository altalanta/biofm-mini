"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Set seeds across libraries and favour deterministic kernels."""

    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("NumPy is required for seeding.") from exc
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for seeding.") from exc

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    try:  # pragma: no cover - depends on torch build capabilities
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - best effort if unsupported
        pass


__all__ = ["seed_everything"]
