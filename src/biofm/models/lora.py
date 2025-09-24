"""Optional LoRA adapters for lightweight fine-tuning."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional speedup
    import bitsandbytes as bnb  # type: ignore

    HAS_BNB = True
except ImportError:  # pragma: no cover - optional speedup
    HAS_BNB = False


class LoRAAdapter(nn.Module):
    """Minimal LoRA wrapper around linear layers."""

    def __init__(self, module: nn.Linear, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.module = module
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_up = nn.Linear(module.in_features, rank, bias=False)
        self.lora_down = nn.Linear(rank, module.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_up.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        original = self.module(x)
        update = self.lora_down(self.lora_up(x)) * self.scaling
        return original + update


def apply_lora_adapters(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int = 4,
    alpha: float = 8.0,
) -> int:
    """Replace matching linear layers with LoRA adapters."""

    replaced = 0
    targets = list(target_modules)
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in targets):
            continue
        parent, attr_name = _resolve_parent(model, name)
        if parent is None:
            continue
        adapter = LoRAAdapter(module, rank=rank, alpha=alpha)
        setattr(parent, attr_name, adapter)
        replaced += 1
    if replaced == 0:
        LOGGER.info("LoRA requested but no matching modules were found; model left unchanged")
    else:
        LOGGER.info("Enabled LoRA on %s modules (bitsandbytes=%s)", replaced, HAS_BNB)
    return replaced


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module | None, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part, None)  # type: ignore[assignment]
        if parent is None:
            return None, ""
    return parent, parts[-1]


__all__ = ["LoRAAdapter", "apply_lora_adapters", "HAS_BNB"]
