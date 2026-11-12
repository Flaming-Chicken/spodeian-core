"""
Continuous Velocity Flow Head.
Predicts the continuous velocity vector field v_theta(x_t, t) for ODE flow integration in RainAI.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal Fourier positional embedding for continuous time t in [0, 1]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FlowVelocityHead(nn.Module):
    """
    Continuous Velocity Field Head.
    Given state x_t in R^D and time t in [0, 1], predicts velocity dx/dt in R^D
    to feed into RainAI's RK4 flow solver.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, state_dim) or (B, T, state_dim)
        t: (B,) or (B, 1) float tensor in [0, 1]
        Returns: (B, state_dim) velocity vector field
        """
        if x.dim() == 3:
            x = x.mean(dim=1)

        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        combined = torch.cat([x, t_emb], dim=-1)
        return self.net(combined)

    def compute_flow_matching_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes flow matching loss: ||v_theta(x_t, t) - (x_1 - x_0)||^2
        where x_t = (1 - t) * x_0 + t * x_1.
        """
        batch_size = x_0.size(0)
        device = x_0.device

        if t is None:
            t = torch.rand(batch_size, 1, device=device)

        # Linear flow interpolation
        x_t = (1.0 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        predicted_velocity = self.forward(x_t, t)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss, predicted_velocity
