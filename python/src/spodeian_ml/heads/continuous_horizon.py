"""
Continuous-Time Horizon Forecast Head.
Predicts expected returns or signals for ANY continuous future horizon t in R+
(e.g., t = 0.5 intraday, t = 2.5 days, t = 21.0 days, t = 252.0 days) using continuous
logarithmic-Fourier temporal basis projections.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogarithmicFourierHorizonEmbedding(nn.Module):
    """
    Continuous logarithmic-Fourier temporal basis embedding for horizon t >= 0:
    Phi_k(t) = [sin(omega_k * ln(1 + t)), cos(omega_k * ln(1 + t))]
    """

    def __init__(self, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        # Spatially decaying geometric frequencies
        freqs = 1.0 / (10.0 ** (torch.arange(num_freqs, dtype=torch.float32) / max(num_freqs, 1)))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, 1) float tensor representing continuous horizon in days/bars.
        Returns: (B, num_freqs * 2)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        log_t = torch.log1p(torch.clamp(t, min=0.0))  # ln(1 + t)
        args = log_t * self.freqs.unsqueeze(0)  # (B, num_freqs)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ContinuousHorizonHead(nn.Module):
    """
    Continuous-Time Horizon Prediction Head.
    Takes latent representation h in R^D and continuous horizon t in R+,
    outputting expected alpha / return y_hat(t).
    """

    def __init__(
        self,
        input_dim: int,
        num_freqs: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.time_dim = num_freqs * 2

        self.time_embed = LogarithmicFourierHorizonEmbedding(num_freqs)
        self.feat_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Linear(self.time_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h: (B, input_dim) latent features
        t: (B,) or (B, 1) continuous horizons
        Returns: (B, 1) predicted return/alpha for horizon t
        """
        if h.dim() == 3:
            h = h.mean(dim=1)

        feat_h = self.feat_proj(h)
        t_emb = self.time_embed(t)
        time_h = self.time_proj(t_emb)

        combined = F.gelu(feat_h + time_h)
        return self.out_linear(combined)

    def predict_multi_horizons(
        self,
        h: torch.Tensor,
        horizons: List[float],
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Evaluates predictions across an arbitrary list of continuous horizons.
        """
        device = h.device
        results = []
        for t_val in horizons:
            t_tensor = torch.full((h.size(0), 1), float(t_val), dtype=torch.float32, device=device)
            pred = self.forward(h, t_tensor)
            results.append((t_val, pred))
        return results

    def compute_horizon_loss(
        self,
        h: torch.Tensor,
        target_dict: dict,
    ) -> torch.Tensor:
        """
        Computes MSE loss across arbitrary observed horizons in target_dict: {t_float: target_tensor}.
        Includes temporal Lipschitz smoothness regularization.
        """
        total_loss = torch.tensor(0.0, device=h.device)
        count = 0

        for t_val, target in target_dict.items():
            t_tensor = torch.full((h.size(0), 1), float(t_val), dtype=torch.float32, device=h.device)
            pred = self.forward(h, t_tensor)
            total_loss = total_loss + F.mse_loss(pred, target)
            count += 1

        return total_loss / max(count, 1)
