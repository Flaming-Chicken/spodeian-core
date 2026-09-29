"""
Learned Neural ODE & Flow Solver Controller.
Trainable adaptive step controller and residual velocity corrector trained end-to-end
with adjoint/backpropagation step-budget regularization.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedFlowSolverController(nn.Module):
    """
    Trainable Neural ODE Step Controller & Velocity Corrector.
    Learns to dynamically modulate step size h_t and residual velocity correction Delta v
    to solve flow trajectories with minimal integration steps while preserving manifold fidelity.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        min_h: float = 0.01,
        max_h: float = 0.25,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.min_h = min_h
        self.max_h = max_h

        # Predicts step scale factor and residual velocity correction
        self.controller = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # [Delta v (state_dim), raw_step_logit (1)]
        )

    def forward(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, state_dim) current state
        v: (B, state_dim) current base flow velocity
        t: (B, 1) current integration time in [0, 1]
        Returns:
            step_size: (B, 1) adaptive step size h_t clamped in [min_h, max_h]
            delta_v: (B, state_dim) residual learned velocity correction
        """
        inputs = torch.cat([x, v, t], dim=-1)
        out = self.controller(inputs)

        delta_v = out[:, : self.state_dim]
        step_logit = out[:, self.state_dim :]
        # Sigmoid maps to [min_h, max_h]
        step_size = self.min_h + (self.max_h - self.min_h) * torch.sigmoid(step_logit)

        return step_size, delta_v

    def solve_trajectory(
        self,
        velocity_net: nn.Module,
        x0: torch.Tensor,
        max_steps: int = 50,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Differentiable ODE integration from t=0.0 to t=1.0 using the learned step controller.
        Returns:
            final_x: (B, state_dim) integrated endpoint at t=1
            trajectory: list of intermediate states
            total_step_penalty: scalar loss penalizing excessive steps
        """
        batch_size = x0.size(0)
        device = x0.device

        current_x = x0
        current_t = torch.zeros((batch_size, 1), device=device)
        trajectory = [current_x]
        step_penalty = torch.tensor(0.0, device=device)

        for _ in range(max_steps):
            mask = current_t < 1.0 - 1e-4
            if not mask.any():
                break

            base_v = velocity_net(current_x, current_t)
            h, delta_v = self.forward(current_x, base_v, current_t)

            # Prevent stepping beyond t=1.0
            h = torch.min(h, 1.0 - current_t)

            effective_v = base_v + 0.1 * delta_v
            current_x = current_x + h * effective_v
            current_t = current_t + h

            trajectory.append(current_x)
            step_penalty = step_penalty + (1.0 / (h + 1e-6)).mean()

        return current_x, trajectory, step_penalty

    def compute_trained_solver_loss(
        self,
        velocity_net: nn.Module,
        x0: torch.Tensor,
        target_x1: torch.Tensor,
        lambda_steps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes joint endpoint trajectory loss + step efficiency penalty.
        """
        final_x, _, step_penalty = self.solve_trajectory(velocity_net, x0)
        endpoint_loss = F.mse_loss(final_x, target_x1)
        total_loss = endpoint_loss + lambda_steps * step_penalty
        return total_loss, endpoint_loss
