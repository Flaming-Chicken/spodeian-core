"""
Rust-accelerated data augmentations for Self-Supervised Learning (SSL) in PyTorch.
Powered by spodeian_py_bridge to guarantee zero Train-Serve Skew with production Rust.
"""

from typing import Union, Sequence
import torch

try:
    from spodeian_ml import spodeian_py_bridge
except ImportError:
    import spodeian_py_bridge


def temporal_mask(
    tensor: Union[torch.Tensor, Sequence[float]],
    mask_ratio: float = 0.2,
    mask_value: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """Applies deterministic random sequence masking via the Rust Spodeian engine."""
    if isinstance(tensor, torch.Tensor):
        device = tensor.device
        dtype = tensor.dtype
        vals = tensor.detach().cpu().view(-1).tolist()
        masked_vals = spodeian_py_bridge.py_temporal_mask(vals, float(mask_ratio), float(mask_value), int(seed))
        return torch.tensor(masked_vals, dtype=dtype, device=device).view_as(tensor)
    else:
        vals = list(tensor)
        masked_vals = spodeian_py_bridge.py_temporal_mask(vals, float(mask_ratio), float(mask_value), int(seed))
        return torch.tensor(masked_vals, dtype=torch.float32)


def gaussian_jitter(
    tensor: Union[torch.Tensor, Sequence[float]],
    std_dev: float = 0.05,
    seed: int = 1234,
) -> torch.Tensor:
    """Applies deterministic Box-Muller Gaussian jitter via the Rust Spodeian engine."""
    if isinstance(tensor, torch.Tensor):
        device = tensor.device
        dtype = tensor.dtype
        vals = tensor.detach().cpu().view(-1).tolist()
        jittered = spodeian_py_bridge.py_gaussian_jitter(vals, float(std_dev), int(seed))
        return torch.tensor(jittered, dtype=dtype, device=device).view_as(tensor)
    else:
        vals = list(tensor)
        jittered = spodeian_py_bridge.py_gaussian_jitter(vals, float(std_dev), int(seed))
        return torch.tensor(jittered, dtype=torch.float32)


def rolling_zscore(
    tensor: Union[torch.Tensor, Sequence[float]],
    window: int = 20,
) -> torch.Tensor:
    """Calculates rolling window Z-Score normalization via the Rust Spodeian engine."""
    if isinstance(tensor, torch.Tensor):
        device = tensor.device
        dtype = tensor.dtype
        vals = tensor.detach().cpu().view(-1).tolist()
        normalized = spodeian_py_bridge.py_rolling_zscore(vals, int(window))
        return torch.tensor(normalized, dtype=dtype, device=device).view_as(tensor)
    else:
        vals = list(tensor)
        normalized = spodeian_py_bridge.py_rolling_zscore(vals, int(window))
        return torch.tensor(normalized, dtype=torch.float32)


def time_warp(
    tensor: Union[torch.Tensor, Sequence[float]],
    target_len: int,
) -> torch.Tensor:
    """Performs linear interpolation time-warping via the Rust Spodeian engine."""
    if isinstance(tensor, torch.Tensor):
        device = tensor.device
        dtype = tensor.dtype
        vals = tensor.detach().cpu().view(-1).tolist()
        warped = spodeian_py_bridge.py_time_warp(vals, int(target_len))
        return torch.tensor(warped, dtype=dtype, device=device)
    else:
        vals = list(tensor)
        warped = spodeian_py_bridge.py_time_warp(vals, int(target_len))
        return torch.tensor(warped, dtype=torch.float32)
