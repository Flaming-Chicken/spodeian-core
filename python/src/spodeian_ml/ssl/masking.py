"""
Universal Sequence and Patch Masking Primitives for Self-Supervised Learning.
Integrates with spodeian_py_bridge for deterministic data corruption.
"""

from typing import Tuple
import torch
import torch.nn as nn
from spodeian_ml.data.augmentations import temporal_mask


class SequenceMasker(nn.Module):
    """Random sequence masking layer for continuous time-series and audio flow tensors."""

    def __init__(self, mask_ratio: float = 0.25, mask_value: float = 0.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes clean tensor (batch_size, seq_len, features) or (batch_size, features).
        Returns (masked_input, mask_boolean_tensor).
        """
        masked = temporal_mask(x, mask_ratio=self.mask_ratio, mask_value=self.mask_value, seed=seed)
        mask_bool = (masked == self.mask_value)
        return masked, mask_bool
