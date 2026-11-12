"""
VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.
Eliminates the requirement for large batches and negative sample mining.
Ideal for smaller financial and audio flow embedding batches.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg loss function (Bardes, Ponce, LeCun - ICLR 2022).
    - Invariance: MSE between representations of two views.
    - Variance: Enforces representation variance >= gamma (prevents constant collapse).
    - Covariance: Decorrelates feature dimensions (prevents dimensional collapse).
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, num_features) view A
        y: (batch_size, num_features) view B
        Returns: (total_loss, sim_loss, std_loss, cov_loss)
        """
        batch_size, num_features = x.size()

        # 1. Invariance Loss (MSE)
        sim_loss = F.mse_loss(x, y)

        # 2. Variance Loss (Standard Deviation Regularization)
        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        std_y = torch.sqrt(y.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.gamma - std_x)) / 2 + torch.mean(F.relu(self.gamma - std_y)) / 2

        # 3. Covariance Loss (Off-diagonal Decorrelation)
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)

        cov_x = (x_centered.T @ x_centered) / (batch_size - 1)
        cov_y = (y_centered.T @ y_centered) / (batch_size - 1)

        # Zero out diagonal
        diag = torch.eye(num_features, device=x.device, dtype=torch.bool)
        cov_loss = (cov_x[~diag].pow(2).sum() / num_features) / 2 + (cov_y[~diag].pow(2).sum() / num_features) / 2

        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return total_loss, sim_loss, std_loss, cov_loss
