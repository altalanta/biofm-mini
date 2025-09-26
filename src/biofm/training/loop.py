"""Simple PyTorch training loop for the CLIP-style model."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from biofm.configuration import AmpOption
from biofm.deps import require_torch
from biofm.training.utils import resolve_amp, save_checkpoint, select_device

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the supervised contrastive training loop."""

    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp_mode: AmpOption = "auto"
    checkpoint_dir: Path = Path("outputs/checkpoints")


@dataclass
class TrainingSummary:
    """Structured record of a completed training run."""

    checkpoint_path: Path
    losses: list[float]


def _autocast_context(torch_module, device_type: str, enabled: bool):
    if hasattr(torch_module, "autocast"):
        return torch_module.autocast(device_type=device_type, enabled=enabled)
    return torch_module.cuda.amp.autocast(device_type=device_type, enabled=enabled)


def train_model(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    config: LoopConfig,
    device: torch.device | None = None,
) -> TrainingSummary:
    """Train the model and capture epoch-level metrics."""

    torch_module = require_torch()
    device = device or select_device()
    model.to(device)
    model.train()

    optimizer = torch_module.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    amp_enabled = resolve_amp(config.amp_mode, device)
    grad_scaler = torch_module.cuda.amp.GradScaler  # type: ignore[attr-defined]
    scaler = grad_scaler(enabled=amp_enabled)
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.pt"

    losses: list[float] = []

    epoch = 0
    try:
        for epoch in range(1, config.epochs + 1):
            epoch_loss = 0.0
            step_count = 0
            for step, batch in enumerate(dataloader, start=1):
                step_count = step
                pixel_values = batch["pixel_values"].to(device)
                expression = batch["expression"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(torch_module, device.type, amp_enabled):
                    outputs = model(pixel_values=pixel_values, expression=expression)
                    loss = outputs["loss"]
                if not torch_module.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch {epoch} step {step}: {loss.item()}"
                    )
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch_module.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    LOGGER.info("Epoch %s Step %s Loss %.4f", epoch, step, loss.item())
            if step_count == 0:
                raise ValueError("Training dataloader yielded no batches")
            save_checkpoint(model, optimizer, last_checkpoint, epoch)
            average_loss = epoch_loss / step_count
            LOGGER.info("Epoch %s average loss %.4f", epoch, average_loss)
            losses.append(average_loss)
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        LOGGER.info("Training interrupted; saving checkpoint before exit")
        if epoch > 0:
            save_checkpoint(model, optimizer, last_checkpoint, epoch)
        raise

    return TrainingSummary(checkpoint_path=last_checkpoint, losses=losses)


__all__ = ["LoopConfig", "TrainingSummary", "train_model"]
