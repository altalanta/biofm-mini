"""Helpers for lazily importing heavy optional dependencies."""

from __future__ import annotations

import importlib
from functools import cache
from typing import Any


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency is required but not installed."""


@cache
def require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - import-time failure
        raise MissingDependencyError(
            "PyTorch is required for this command. Install biofm[gpu] or add torch manually."
        ) from exc


@cache
def require_torchvision() -> Any:
    try:
        return importlib.import_module("torchvision")
    except Exception as exc:  # pragma: no cover - import-time failure
        raise MissingDependencyError(
            "torchvision is required for image encoders. Install biofm[gpu] or add torchvision manually."
        ) from exc


def require_torch_transforms() -> Any:
    torchvision = require_torchvision()
    try:
        return torchvision.transforms
    except AttributeError as exc:  # pragma: no cover - defensive
        raise MissingDependencyError("torchvision.transforms is unavailable") from exc


def require_torch_data() -> Any:
    torch = require_torch()
    try:
        return torch.utils.data
    except AttributeError as exc:  # pragma: no cover - defensive
        raise MissingDependencyError("torch.utils.data is unavailable") from exc


__all__ = [
    "MissingDependencyError",
    "require_torch",
    "require_torchvision",
    "require_torch_transforms",
    "require_torch_data",
]
