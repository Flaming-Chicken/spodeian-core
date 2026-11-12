"""
Regime Classification & Symbolic Decision Head.
Used for market regime detection (Bull, Bear, Crisis, Low-Vol Chop) in Risky Business,
and discrete proof/tactic branch selection in algebra-engine.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeClassificationHead(nn.Module):
    """
    Discrete classification head with temperature calibration and label smoothing.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unnormalized logits."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x) / self.temperature

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized probability distribution over discrete classes."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        targets: (B,) class indices or (B, num_classes) one-hot distribution.
        Returns (cross_entropy_loss, logits).
        """
        logits = self.forward(x)
        loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        return loss, logits
