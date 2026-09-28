"""
Bit-Perfect Deterministic Checkpoint Engine for spodeian-ml.
Captures and restores model weights, AdamW optimizer momentum buffers,
learning rate scheduler curves, step counters, and multi-source PRNG seeds.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    epoch: int,
    global_step: int,
    best_loss: float,
    config_dict: Dict[str, Any],
) -> Path:
    """Saves a complete state dictionary allowing bit-perfect resumption of training."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "config": config_dict,
        # Save PRNG seeds for deterministic resumption of data shuffling and augmentations
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }

    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()

    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """
    Restores the exact training environment from disk.
    Returns (start_epoch, global_step, best_loss, config_dict).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    target_device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    # 1. Restore Model Parameters
    model.load_state_dict(checkpoint["model_state"])

    # 2. Restore Optimizer Momentum & Variance Buffers
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    # 3. Restore Scheduler
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    # 4. Restore Random Number Generators
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
    if "numpy_rng" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng"])
    if "python_rng" in checkpoint:
        random.setstate(checkpoint["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))
    config = checkpoint.get("config", {})

    return epoch, global_step, best_loss, config
