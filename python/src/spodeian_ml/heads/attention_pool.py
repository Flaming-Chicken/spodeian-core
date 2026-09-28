"""
Attention Pooling Head.
Eliminates naive uniform mean-pooling by learning a temporal attention query
that weights critical catalyst moments (volatility spikes, breakout ticks) across the sequence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolingHead(nn.Module):
    """
    Self-Attention temporal pooling head.
    Given sequence tensor (B, T, D), weights tokens via learned scoring vector:
    alpha = Softmax(tanh(x @ W) @ v)
    pooled = sum(alpha * x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        attn_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.attn_proj = nn.Linear(in_features, attn_dim)
        self.attn_vector = nn.Linear(attn_dim, 1, bias=False)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D (batch, features), treat as single time-step
        if x.dim() == 2:
            return self.output_fc(self.norm(x))

        # x: (B, T, D)
        # Compute attention scores over sequence dimension T
        proj = torch.tanh(self.attn_proj(x))  # (B, T, attn_dim)
        scores = self.attn_vector(proj)       # (B, T, 1)
        weights = F.softmax(scores, dim=1)    # (B, T, 1)

        # Weighted sum pooling
        pooled = torch.sum(x * weights, dim=1)  # (B, D)
        pooled = self.dropout(self.norm(pooled))
        return self.output_fc(pooled)
