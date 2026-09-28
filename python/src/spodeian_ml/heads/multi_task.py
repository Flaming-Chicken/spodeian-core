"""
Dynamic Multi-Horizon & Multi-Task Prediction Head.
Dynamically supports ANY combination of forecast horizons (e.g., 1-day, 2-day, 5-day, 10-day,
21-day, 63-day, 252-day annual returns) using Kendall & Gal dynamic homoscedastic uncertainty weighting.
"""

from typing import Dict, List, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_horizon_name(horizon: Union[int, str]) -> Tuple[str, int]:
    """Normalizes an integer or string horizon into (key_name, day_count)."""
    if isinstance(horizon, int):
        if horizon <= 0:
            raise ValueError(f"Horizon days must be positive integers, got: {horizon}")
        return f"h{horizon}", horizon
    elif isinstance(horizon, str):
        cleaned = horizon.strip().lower()
        if cleaned.startswith("h") and cleaned[1:].isdigit():
            return cleaned, int(cleaned[1:])
        elif cleaned.isdigit():
            return f"h{cleaned}", int(cleaned)
        else:
            return cleaned, 1
    else:
        raise TypeError(f"Unsupported horizon type: {type(horizon)}")


class MultiHorizonHead(nn.Module):
    """
    Fully dynamic multi-horizon head for financial return and risk term-structure modeling.
    Accepts arbitrary sequences of forecast horizons (e.g., [1, 2, 5, 10, 21, 63, 252]).
    """

    def __init__(
        self,
        in_features: int,
        horizons: Sequence[Union[int, str]] = (1, 5, 21),
        hidden_dim: int = 64,
        out_dim_per_task: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_dim_per_task = out_dim_per_task

        # Parse and sort horizons by day count
        parsed = [normalize_horizon_name(h) for h in horizons]
        parsed.sort(key=lambda x: x[1])

        self.horizon_keys: List[str] = [p[0] for p in parsed]
        self.horizon_days: List[int] = [p[1] for p in parsed]
        self.num_horizons = len(self.horizon_keys)

        self.heads = nn.ModuleDict()
        self.log_vars = nn.ParameterDict()

        for key in self.horizon_keys:
            self.heads[key] = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim_per_task),
            )
            # Initialize log variance to 0.0 (sigma = 1.0)
            self.log_vars[key] = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dictionary of predictions keyed by horizon name."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        return {key: self.heads[key](x) for key in self.horizon_keys}

    def forward_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions concatenated into a continuous (batch_size, num_horizons) matrix."""
        preds_dict = self.forward(x)
        tensors = [preds_dict[k] for k in self.horizon_keys]
        return torch.cat(tensors, dim=-1)

    def compute_loss(
        self,
        predictions: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: Union[Dict[Union[str, int], torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Dynamically computes homoscedastic uncertainty loss across arbitrary horizons.
        Accepts targets as either a dictionary or a tensor matrix (B, num_horizons).
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        losses_per_task: Dict[str, float] = {}

        # Convert tensor targets to dictionary if matrix is provided
        if isinstance(targets, torch.Tensor):
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            target_dict = {
                self.horizon_keys[i]: targets[:, i : i + 1]
                for i in range(min(targets.size(1), self.num_horizons))
            }
        else:
            # Normalize dictionary keys
            target_dict = {}
            for k, v in targets.items():
                norm_key, _ = normalize_horizon_name(k)
                target_dict[norm_key] = v

        # Convert predictions to dictionary if matrix is provided
        if isinstance(predictions, torch.Tensor):
            pred_dict = {
                self.horizon_keys[i]: predictions[:, i : i + 1]
                for i in range(min(predictions.size(1), self.num_horizons))
            }
        else:
            pred_dict = predictions

        for key in self.horizon_keys:
            if key in target_dict and key in pred_dict:
                pred = pred_dict[key]
                target = target_dict[key]
                mse = F.mse_loss(pred, target)
                s = self.log_vars[key]

                task_loss = 0.5 * torch.exp(-s) * mse + 0.5 * s
                total_loss = total_loss + task_loss.squeeze()
                losses_per_task[key] = mse.item()

        return total_loss, losses_per_task
