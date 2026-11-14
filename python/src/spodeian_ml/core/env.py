"""Deterministic environment setup and device capability detection for AI-Examples."""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EnvInfo:
    device: torch.device
    seed: int
    supports_bf16: bool
    supports_fp8: bool
    torch_version: str

    def __str__(self) -> str:
        return (
            f"EnvInfo(device={self.device.type}, seed={self.seed}, "
            f"bf16={self.supports_bf16}, fp8={self.supports_fp8}, torch={self.torch_version})"
        )


def setup_env(seed: int = 42, prefer_device: str | None = None) -> EnvInfo:
    """Configures deterministic execution and detects hardware capabilities.

    Args:
        seed: Random seed for PyTorch, NumPy, and Python RNGs.
        prefer_device: Specific device override ('cpu', 'cuda', 'mps'). Defaults to auto-detection.

    Returns:
        EnvInfo containing the resolved device and supported data types.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(".cache/torchinductor")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if prefer_device is not None:
        device = torch.device(prefer_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    supports_bf16 = False
    if device.type == "cuda":
        supports_bf16 = torch.cuda.is_bf16_supported()
    elif device.type == "cpu":
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16) + torch.ones(1, dtype=torch.bfloat16)
            supports_bf16 = True
        except RuntimeError:
            supports_bf16 = False

    supports_fp8 = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")

    return EnvInfo(
        device=device,
        seed=seed,
        supports_bf16=supports_bf16,
        supports_fp8=supports_fp8,
        torch_version=torch.__version__,
    )
