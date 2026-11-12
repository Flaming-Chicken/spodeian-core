"""
Linear Probe Evaluation Head.
Used for rapid zero-cost validation of frozen Self-Supervised Learning (SSL) representations.
"""

import torch
import torch.nn as nn


class LinearProbeHead(nn.Module):
    """Single linear projection head with optional batch/layer normalization."""

    def __init__(self, in_features: int, out_features: int = 1, use_layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 3D (batch, seq, feat), take pooled representation or last token
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self.norm(x)
        return self.linear(x)
