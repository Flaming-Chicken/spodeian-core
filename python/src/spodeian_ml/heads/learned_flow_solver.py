"""
Learned Neural ODE & Flow Solver Controller.
Trainable adaptive step controller and residual velocity corrector trained end-to-end
with adjoint/backpropagation step-budget regularization.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_torch_precision_guards(dtype: torch.dtype) -> Tuple[float, float, float]:
    """Returns (denom_guard, temporal_guard, energy_drift_guard) for a given torch dtype."""
    if dtype == torch.float64:
        return 1e-15, 1e-14, 2e-15
    elif dtype == torch.float32:
        return 1e-8, 1e-6, 1e-7
    elif dtype == torch.bfloat16:
        return 1e-3, 1e-2, 8e-3
    elif dtype == torch.float16:
        return 1e-4, 1e-3, 1e-3
    else:
        eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-7
        return eps * 0.1, eps * 10.0, eps


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

    def solve_gated_trajectory(
        self,
        velocity_net: nn.Module,
        x0: torch.Tensor,
        max_steps: int = 50,
        tau_pe: float = 0.02,
        tau_pec: float = 0.08,
        tau_pece: float = 0.20,
        lambda_pareto: Optional[float] = None,
        eval_latency_sec: float = 1e-4,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        """
        Differentiable ODE integration with 4-tier granular dynamic evaluation gating:
        PE <-> PEC <-> PECE <-> PECEC.
        
        - PE (1 NFE): Predictor output accepted directly on ultra-smooth trajectories.
        - PEC (1 NFE): Corrector transformation applied to state using predictor velocity; 0 extra evaluations!
        - PECE (2 NFE): Velocity re-evaluated at corrected state to refresh derivative accuracy.
        - PECEC (2 NFE): Secondary corrector refinement applied for stiff boundary layers.
        
        Optionally applies Information-Theoretic Gating when `lambda_pareto` is provided:
        stopping secondary evaluations when information rate (bits/sec) < lambda_pareto.
        
        Returns:
            final_x: (B, state_dim)
            trajectory: list of states
            metrics: dict with total NFE and tier counts (PE, PEC, PECE, PECEC)
        """
        batch_size = x0.size(0)
        device = x0.device
        denom_guard, temporal_guard, _ = get_torch_precision_guards(x0.dtype)

        current_x = x0
        current_t = torch.zeros((batch_size, 1), device=device, dtype=x0.dtype)
        trajectory = [current_x]
        total_nfe = 0
        pe_count = 0
        pec_count = 0
        pece_count = 0
        pecec_count = 0

        for _ in range(max_steps):
            mask = current_t < 1.0 - temporal_guard
            if not mask.any():
                break

            # 1. Base velocity and step size
            base_v = velocity_net(current_x, current_t)
            h, delta_v = self.forward(current_x, base_v, current_t)
            h = torch.min(h, 1.0 - current_t)

            # 2. Predictor Step (P)
            x_pred = current_x + h * (base_v + 0.1 * delta_v)

            # 3. Evaluate at predicted state (E)
            v_pred = velocity_net(x_pred, current_t + h)
            total_nfe += 1

            # Estimate scale-invariant displacement curvature metric
            displacement = (torch.norm(x_pred - current_x, p=2, dim=-1) / (torch.norm(current_x, p=2, dim=-1) + denom_guard)).mean().item()

            if displacement < tau_pe:
                # Tier 1: PE (1 NFE)
                current_x = x_pred
                pe_count += 1
            else:
                # Corrector calculation (C) using v_pred
                x_corr = current_x + 0.5 * h * (base_v + v_pred + 0.1 * delta_v)
                
                # Check Information-Theoretic Pareto Stopping Policy if active
                stop_at_pec = False
                if lambda_pareto is not None:
                    diff_norm = torch.norm(x_corr - x_pred, p=2, dim=-1).mean().item()
                    info_gain_bits = torch.log2(torch.tensor(1.0 + diff_norm / 1e-4)).item()
                    marginal_rate = info_gain_bits / max(eval_latency_sec, 1e-9)
                    if marginal_rate < lambda_pareto:
                        stop_at_pec = True
                elif displacement < tau_pec:
                    stop_at_pec = True

                if stop_at_pec:
                    # Tier 2: PEC (1 NFE, corrected state with zero extra evaluations)
                    current_x = x_corr
                    pec_count += 1
                else:
                    # Second evaluation (E) at corrected state
                    v_corr = velocity_net(x_corr, current_t + h)
                    total_nfe += 1
                    if displacement < tau_pece:
                        # Tier 3: PECE (2 NFE, derivative re-evaluated for accuracy)
                        current_x = x_corr
                        pece_count += 1
                    else:
                        # Tier 4: PECEC (2 NFE, double corrector refinement for stiff transients)
                        x_refined = current_x + 0.5 * h * (base_v + v_corr + 0.1 * delta_v)
                        current_x = x_refined
                        pecec_count += 1

            current_t = current_t + h
            trajectory.append(current_x)

        metrics = {
            "total_nfe": total_nfe,
            "pe_steps": pe_count,
            "pec_steps": pec_count,
            "pece_steps": pece_count,
            "pecec_steps": pecec_count,
        }
        return current_x, trajectory, metrics

    def compute_trained_solver_loss(
        self,
        velocity_net: nn.Module,
        x0: torch.Tensor,
        target_x1: torch.Tensor,
        lambda_steps: float = 1e-3,
        hamiltonian_fn: Optional[nn.Module] = None,
        lambda_hamiltonian: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes joint endpoint trajectory loss + step efficiency penalty + optional Hamiltonian energy conservation loss.
        """
        final_x, trajectory, step_penalty = self.solve_trajectory(velocity_net, x0)
        endpoint_loss = F.mse_loss(final_x, target_x1)
        total_loss = endpoint_loss + lambda_steps * step_penalty

        if hamiltonian_fn is not None and lambda_hamiltonian > 0.0:
            e0 = hamiltonian_fn(x0)
            energy_loss = torch.tensor(0.0, device=x0.device)
            for xt in trajectory[1:]:
                energy_loss = energy_loss + F.mse_loss(hamiltonian_fn(xt), e0)
            energy_loss = energy_loss / max(1, len(trajectory) - 1)
            total_loss = total_loss + lambda_hamiltonian * energy_loss

        return total_loss, endpoint_loss
