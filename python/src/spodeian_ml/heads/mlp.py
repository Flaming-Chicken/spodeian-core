"""
Multi-Layer Perceptron (MLP) Projection Head.
Used for non-linear mapping of latent representations to quantitative alpha signals and classifications.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class MLPProjectionHead(nn.Module):
    """Deep non-linear projection head with LayerNorm, GELU/SiLU, and Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        out_features: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        dims = hidden_dims or [in_features, in_features // 2]
        layers = []
        curr = in_features

        for h in dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.LayerNorm(h))
            if activation.lower() == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr = h

        layers.append(nn.Linear(curr, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)
