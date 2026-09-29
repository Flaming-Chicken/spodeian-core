"""
Tests for ContinuousHorizonHead and LearnedFlowSolverController.
Verifies continuous-time horizon projections, Fourier temporal continuity,
and end-to-end differentiable ODE flow trajectory solving with learned step adaptation.
"""

import pytest
import torch
import torch.nn as nn
from spodeian_ml.heads.continuous_horizon import ContinuousHorizonHead, LogarithmicFourierHorizonEmbedding
from spodeian_ml.heads.learned_flow_solver import LearnedFlowSolverController
from spodeian_ml.heads.flow import FlowVelocityHead


def test_continuous_horizon_head_projections():
    batch_size = 4
    input_dim = 16
    head = ContinuousHorizonHead(input_dim=input_dim, num_freqs=6, hidden_dim=32)

    h = torch.randn(batch_size, input_dim)
    t = torch.tensor([0.5, 1.0, 7.0, 21.0])

    pred = head(h, t)
    assert pred.shape == (batch_size, 1)
    assert not torch.isnan(pred).any()

    # Multi-horizon evaluation
    horizons = [0.25, 1.5, 5.0, 10.0, 21.3, 63.0, 252.0]
    multi_preds = head.predict_multi_horizons(h, horizons)
    assert len(multi_preds) == len(horizons)
    for t_val, p in multi_preds:
        assert p.shape == (batch_size, 1)

    # Continuity check: t and t + eps produce close predictions
    t1 = torch.tensor([5.0])
    t2 = torch.tensor([5.001])
    h_single = torch.randn(1, input_dim)
    p1 = head(h_single, t1)
    p2 = head(h_single, t2)
    assert torch.abs(p1 - p2).item() < 1e-2


def test_learned_flow_solver_controller_trajectory():
    state_dim = 4
    velocity_head = FlowVelocityHead(state_dim=state_dim, hidden_dim=32, time_embed_dim=16)
    solver = LearnedFlowSolverController(state_dim=state_dim, hidden_dim=32, min_h=0.05, max_h=0.25)

    x0 = torch.randn(2, state_dim)
    target_x1 = torch.randn(2, state_dim)

    # 1. Forward single step
    v = velocity_head(x0, torch.zeros(2, 1))
    step_size, delta_v = solver(x0, v, torch.zeros(2, 1))
    assert step_size.shape == (2, 1)
    assert delta_v.shape == (2, state_dim)
    assert (step_size >= solver.min_h).all()
    assert (step_size <= solver.max_h).all()

    # 2. Differentiable trajectory integration
    final_x, trajectory, step_penalty = solver.solve_trajectory(velocity_head, x0, max_steps=30)
    assert final_x.shape == (2, state_dim)
    assert len(trajectory) > 1
    assert step_penalty.item() >= 0.0

    # 3. Trainability: backward pass computes gradients for both velocity_head and solver
    total_loss, endpoint_loss = solver.compute_trained_solver_loss(velocity_head, x0, target_x1)
    total_loss.backward()

    for param in solver.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

    for param in velocity_head.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
