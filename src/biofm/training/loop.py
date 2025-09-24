"""Simple PyTorch training loop for the CLIP-style model."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from biofm.training.utils import get_device, save_checkpoint

LOGGER = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    checkpoint_dir: Path = Path("outputs/checkpoints")


def train_model(
    model: nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    config: LoopConfig,
    device: torch.device | None = None,
) -> Path:
    """Train the model and return path to the last checkpoint."""

    device = device or get_device()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    last_checkpoint = config.checkpoint_dir / "last.pt"

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_loss = 0.0
            step_count = 0
            for step, batch in enumerate(dataloader, start=1):
                step_count = step
                try:
                    pixel_values = batch["pixel_values"].to(device)
                    expression = batch["expression"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                        outputs = model(pixel_values=pixel_values, expression=expression)
                        loss = outputs["loss"]
                        
                    # Check for invalid loss values
                    if not torch.isfinite(loss):
                        LOGGER.warning("Invalid loss detected at epoch %s step %s, skipping", epoch, step)
                        continue
                        
                    scaler.scale(loss).backward()
                    if config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    if step % 10 == 0:
                        LOGGER.info("Epoch %s Step %s Loss %.4f", epoch, step, loss.item())
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        LOGGER.error("GPU out of memory at epoch %s step %s", epoch, step)
                        torch.cuda.empty_cache()
                        raise
                    else:
                        LOGGER.error("Runtime error at epoch %s step %s: %s", epoch, step, e)
                        raise
                        
            save_checkpoint(model, optimizer, last_checkpoint, epoch)
            average_loss = epoch_loss / max(step_count, 1)
            LOGGER.info("Epoch %s average loss %.4f", epoch, average_loss)
            
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user, saving checkpoint...")
        save_checkpoint(model, optimizer, last_checkpoint, epoch)
        
    return last_checkpoint


__all__ = ["LoopConfig", "train_model"]
