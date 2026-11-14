"""Deterministic procedural micro-dataset generators across all 8 scientific domains."""

from __future__ import annotations

import math

import torch


def micro_tabular(
    n_samples: int = 64,
    n_features: int = 8,
    task: str = "regression",
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic tabular regression or binary classification micro-data."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, n_features, generator=gen)
    true_weights = torch.randn(n_features, 1, generator=gen)
    y_linear = x @ true_weights + noise * torch.randn(n_samples, 1, generator=gen)
    if task == "classification":
        y = (y_linear > 0).float()
    else:
        y = y_linear
    return x, y


def micro_image_patches(
    batch_size: int = 2,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    patch_size: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Generates synthetic multi-channel image patches [B, N_patches, channels * patch_size^2]."""
    gen = torch.Generator().manual_seed(seed)
    images = torch.rand(batch_size, channels, height, width, generator=gen)
    # Unfold into non-overlapping patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Shape: [B, C, H_p, W_p, P, P] -> [B, H_p * W_p, C * P * P]
    h_p, w_p = patches.shape[2], patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, h_p * w_p, channels * patch_size * patch_size)
    return patches


def micro_audio_waveform(
    batch_size: int = 2,
    n_samples: int = 16000,
    sample_rate: int = 16000,
    seed: int = 42,
) -> torch.Tensor:
    """Generates synthetic multi-harmonic acoustic audio waveforms [B, n_samples]."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, n_samples / sample_rate, n_samples)
    waveforms = torch.zeros(batch_size, n_samples, dtype=torch.float32)

    for b in range(batch_size):
        f0 = 220.0 + 110.0 * float(torch.rand(1, generator=gen).item())
        # Fundamental + 3 harmonics with decaying amplitude
        signal = torch.sin(2.0 * math.pi * f0 * t)
        signal += 0.5 * torch.sin(2.0 * math.pi * 2.0 * f0 * t)
        signal += 0.25 * torch.sin(2.0 * math.pi * 3.0 * f0 * t)
        # Add light noise
        signal += 0.05 * torch.randn(n_samples, generator=gen)
        # Normalize to [-1, 1]
        max_val = torch.max(torch.abs(signal))
        waveforms[b] = signal / (max_val + 1e-8)

    return waveforms


def micro_control_trajectories(
    batch_size: int = 8,
    horizon: int = 20,
    state_dim: int = 4,
    action_dim: int = 2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates smooth state-action rollout sequences for control/RL benchmarks."""
    gen = torch.Generator().manual_seed(seed)
    actions = torch.tanh(torch.randn(batch_size, horizon, action_dim, generator=gen))
    states = torch.empty((batch_size, horizon, state_dim), dtype=torch.float32)
    s = torch.zeros(batch_size, state_dim)
    for t in range(horizon):
        a = actions[:, t, :]
        a_proj = torch.nn.functional.pad(a, (0, state_dim - action_dim))
        s = 0.95 * s + 0.1 * a_proj + 0.02 * torch.randn(batch_size, state_dim, generator=gen)
        states[:, t, :] = s
    return states, actions


def micro_burgers(
    n_spatial: int = 64,
    n_temporal: int = 32,
    nu: float = 0.01,
    dt: float = 0.001,
) -> torch.Tensor:
    """Solves 1D viscous Burgers' PDE on domain [0, 1] using finite differences."""
    dx = 1.0 / n_spatial
    x = torch.linspace(0.0, 1.0, n_spatial, dtype=torch.float32)
    u = torch.sin(2.0 * math.pi * x)
    solution = torch.empty(n_temporal, n_spatial, dtype=torch.float32)
    solution[0] = u

    for t in range(1, n_temporal):
        u_prev = solution[t - 1]
        du_dx = (torch.roll(u_prev, -1) - torch.roll(u_prev, 1)) / (2.0 * dx)
        d2u_dx2 = (torch.roll(u_prev, -1) - 2.0 * u_prev + torch.roll(u_prev, 1)) / (dx**2)
        u_next = u_prev + dt * (-u_prev * du_dx + nu * d2u_dx2)
        solution[t] = u_next
    return solution


def micro_lorenz(
    n_steps: int = 128,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generates chaotic Lorenz attractor trajectory using RK4 integration."""
    gen = torch.Generator().manual_seed(seed)
    traj = torch.empty(n_steps, 3, dtype=torch.float32)
    state = torch.tensor([1.0, 1.0, 1.0]) + 0.1 * torch.randn(3, generator=gen)
    traj[0] = state

    def f(s: torch.Tensor) -> torch.Tensor:
        dx = sigma * (s[1] - s[0])
        dy = s[0] * (rho - s[2]) - s[1]
        dz = s[0] * s[1] - beta * s[2]
        return torch.stack([dx, dy, dz])

    for i in range(1, n_steps):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i] = state
    return traj


def micro_protein_frames(
    n_residues: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rigid residue backbone frames (Rotations R in SO(3) [N, 3, 3], Translations t in R^3 [N, 3])."""
    gen = torch.Generator().manual_seed(seed)
    # Generate sequential C_alpha translation coordinates along an alpha-helix curve
    t_vals = torch.linspace(0, 4.0 * math.pi, n_residues)
    x = 5.0 * torch.cos(t_vals)
    y = 5.0 * torch.sin(t_vals)
    z = 1.5 * t_vals
    translations = torch.stack([x, y, z], dim=-1)

    # Random valid SO(3) orthogonal rotation matrices via Gram-Schmidt
    raw_m = torch.randn(n_residues, 3, 3, generator=gen)
    q, r = torch.linalg.qr(raw_m)
    # Ensure determinant = +1
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-2)
    rotations = q

    return rotations, translations


def micro_multivariate_timeseries(
    batch_size: int = 4,
    horizon: int = 64,
    n_channels: int = 6,
    trend: bool = True,
    seasonality: bool = True,
    seed: int = 42,
) -> torch.Tensor:
    """Generates multivariate financial/physical time-series with trend and seasonality [B, H, C]."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 1.0, horizon).view(1, horizon, 1)
    series = torch.zeros(batch_size, horizon, n_channels)

    if trend:
        slopes = torch.randn(batch_size, 1, n_channels, generator=gen) * 2.0
        series += slopes * t

    if seasonality:
        freqs = torch.tensor([2.0, 5.0, 10.0, 15.0, 20.0, 25.0])[:n_channels]
        freqs = freqs.view(1, 1, -1)
        series += torch.sin(2.0 * math.pi * freqs * t)

    # Correlated Gaussian noise
    cov = torch.eye(n_channels) * 0.1
    noise = torch.randn(batch_size, horizon, n_channels, generator=gen) @ cov
    series += noise
    return series


def micro_tokens(
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    """Generates synthetic integer token IDs for autoregressive and embedding tests."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)


def micro_raw_bytes(
    batch_size: int = 2,
    length: int = 128,
    seed: int = 42,
) -> torch.Tensor:
    """Generates synthetic raw uint8 byte sequences for tokenizer-free byte models [B, length]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch_size, length), dtype=torch.uint8, generator=gen)


def micro_particles(
    n_particles: int = 64,
    n_dim: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic particle positions and velocities in bounded space."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randn(n_particles, n_dim, generator=gen)
    vel = 0.2 * torch.randn(n_particles, n_dim, generator=gen)
    return pos, vel


def micro_graph(
    n_nodes: int = 32,
    p_edge: float = 0.15,
    feature_dim: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates an Erdős-Rényi random graph (edge_index [2, E], node features [N, F])."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, feature_dim, generator=gen)
    adj_prob = torch.rand(n_nodes, n_nodes, generator=gen)
    adj = (adj_prob < p_edge) & ~torch.eye(n_nodes, dtype=torch.bool)
    adj = adj | adj.T
    edge_index = adj.nonzero().t().contiguous()
    return edge_index, x


def micro_causal_scm(
    n_samples: int = 128,
    n_nodes: int = 6,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic data from a linear-Gaussian Structural Causal Model DAG."""
    gen = torch.Generator().manual_seed(seed)
    adj = torch.triu(torch.randn(n_nodes, n_nodes, generator=gen) * 0.7, diagonal=1)
    noise = torch.randn(n_samples, n_nodes, generator=gen) * 0.5
    identity = torch.eye(n_nodes)
    inv_matrix = torch.linalg.inv(identity - adj.t())
    data = noise @ inv_matrix.t()
    return data, adj
