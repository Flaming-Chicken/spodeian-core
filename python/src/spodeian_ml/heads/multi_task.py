"""
Multi-Horizon & Multi-Task Prediction Head.
Predicts multiple forecast horizons (e.g., 1-day, 5-day, 20-day returns) simultaneously
using Kendall & Gal dynamic homoscedastic uncertainty weighting.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHorizonHead(nn.Module):
    """
    Multi-task head with learnable task uncertainty weights.
    Balances loss contributions across multiple horizons without manual hyperparameter tuning.
    """

    def __init__(
        self,
        in_features: int,
        horizons: List[str] = ("h1", "h5", "h21"),
        hidden_dim: int = 64,
        out_dim_per_task: int = 1,
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.heads = nn.ModuleDict()

        # Learnable log variances s = ln(sigma^2) for uncertainty weighting
        self.log_vars = nn.ParameterDict()

        for h in self.horizons:
            self.heads[h] = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim_per_task),
            )
            # Initialize log variance to 0.0 (sigma = 1.0)
            self.log_vars[h] = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.mean(dim=1)

        return {h: self.heads[h](x) for h in self.horizons}

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes dynamic homoscedastic uncertainty loss across all horizon targets:
        loss = sum_i (0.5 * exp(-s_i) * MSE_i + 0.5 * s_i)
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        losses_per_task = {}

        for h in self.horizons:
            if h in targets:
                pred = predictions[h]
                target = targets[h]
                mse = F.mse_loss(pred, target)
                s = self.log_vars[h]

                task_loss = 0.5 * torch.exp(-s) * mse + 0.5 * s
                total_loss = total_loss + task_loss.squeeze()
                losses_per_task[h] = mse.item()

        return total_loss, losses_per_task
