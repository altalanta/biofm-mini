"""Tiny registry utility to decouple config from code."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._objects: dict[str, T] = {}

    def register(self, name: str, obj: T) -> None:
        if name in self._objects:
            raise ValueError(f"Object named {name} already registered")
        self._objects[name] = obj

    def get(self, name: str) -> T:
        if name not in self._objects:
            available = ", ".join(sorted(self._objects))
            raise KeyError(f"Unknown object {name}. Available: {available}")
        return self._objects[name]

    def names(self) -> Iterable[str]:
        return self._objects.keys()

    def maybe_get(self, name: str) -> T | None:
        return self._objects.get(name)


__all__ = ["Registry"]
