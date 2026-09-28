"""
Quantile Regression & Value-at-Risk (VaR) Head.
Predicts non-parametric distribution quantiles (e.g., 5th, 50th, 95th percentiles)
using asymmetric Pinball Loss for quantitative financial risk management in Risky Business.
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """
    Computes average Pinball (Quantile) Loss across specified quantiles.
    predictions: (batch_size, num_quantiles)
    targets: (batch_size, 1) or (batch_size,)
    """
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    errors = targets - predictions  # (B, Q)
    losses = []

    for i, q in enumerate(quantiles):
        err = errors[:, i]
        loss_q = torch.max(q * err, (q - 1.0) * err)
        losses.append(loss_q.mean())

    return torch.stack(losses).mean()


class QuantileHead(nn.Module):
    """
    Multi-quantile regression head with monotonicity preservation.
    Outputs Value-at-Risk (tail risk), median, and upside potential simultaneously.
    """

    def __init__(
        self,
        in_features: int,
        quantiles: Sequence[float] = (0.01, 0.05, 0.50, 0.95, 0.99),
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)

    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (quantile_loss, predictions)."""
        preds = self.forward(x)
        loss = pinball_loss(preds, targets, self.quantiles)
        return loss, preds
