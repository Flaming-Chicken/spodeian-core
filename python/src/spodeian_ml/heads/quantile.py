"""
Dynamic Quantile Regression & Value-at-Risk (VaR) Head.
Dynamically supports ANY arbitrary combination of quantiles (e.g., 1%, 5%, 25%, 50%, 75%, 95%, 99%)
with optional non-crossing monotonicity enforcement to guarantee q_1 <= q_2 <= ... <= q_k.
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    """
    Computes average Pinball (Quantile) Loss across arbitrary specified quantiles.
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
    Fully dynamic quantile regression head.
    Supports arbitrary custom quantile sequences with optional mathematical monotonicity guarantee.
    """

    def __init__(
        self,
        in_features: int,
        quantiles: Sequence[float] = (0.01, 0.05, 0.50, 0.95, 0.99),
        hidden_dim: int = 64,
        enforce_monotonic: bool = True,
    ):
        super().__init__()
        # Validate and sort quantiles strictly in (0.0, 1.0)
        sorted_quantiles = sorted(list(set(quantiles)))
        for q in sorted_quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"Quantiles must be strictly in (0.0, 1.0), got: {q}")

        self.quantiles = sorted_quantiles
        self.num_quantiles = len(self.quantiles)
        self.enforce_monotonic = enforce_monotonic

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fc = nn.Linear(hidden_dim, self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)

        features = self.backbone(x)
        raw_preds = self.fc(features)  # (B, Q)

        if not self.enforce_monotonic or self.num_quantiles <= 1:
            return raw_preds

        # Enforce non-crossing: q_0 is base, subsequent quantiles are cumulative Softplus increments
        base = raw_preds[:, 0:1]
        increments = F.softplus(raw_preds[:, 1:])  # Strictly non-negative
        cumulative = torch.cumsum(increments, dim=1)
        monotonic_preds = torch.cat([base, base + cumulative], dim=1)
        return monotonic_preds

    def compute_loss(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pinball_loss, predictions)."""
        preds = self.forward(x)
        loss = pinball_loss(preds, targets, self.quantiles)
        return loss, preds
