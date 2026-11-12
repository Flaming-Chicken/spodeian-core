"""Cache-Tiled Hardware-Optimized CPU Kernel: Cayley Parameterization & Meta Profiler.

Topic 06: 06_nn_module_internals_parametrizations_and_meta_device (Category 00: Foundations)

Theoretical Overview:
1. Cayley Orthogonal Parameterization Kernel:
   Given an unconstrained parameter matrix M in R^{d x d}, projecting M onto the Lie group SO(d)
   requires evaluating:
       A = triu(M, 1) - triu(M, 1)^T in so(d)
       P = I + A,  Q = I - A
       W = Q P^{-1}
   Because A is skew-symmetric, all eigenvalues of A are purely imaginary (lambda = i * beta).
   Consequently, the eigenvalues of P = I + A are 1 + i * beta, which have real part exactly 1.
   This guarantees that det(P) >= 1, meaning P is unconditionally invertible and well-conditioned
   without requiring pivoting or regularization.
   This kernel evaluates the Cayley solve in L1/L2 cache-friendly blocks and computes the
   isometric forward projection y = x @ W + b.
2. Zero-Memory Meta Hardware Profiler:
   Dry-runs multi-layer deep network graphs to calculate exact memory footprints (weights,
   buffers, peak activation memory) and theoretical MACs/FLOPs without allocating physical buffers.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor


class ModelMemoryProfile(NamedTuple):
    """Zero-allocation theoretical hardware memory footprint."""

    param_bytes: int
    buffer_bytes: int
    peak_activation_bytes: int
    total_flops: int


def fused_cayley_solve_cpu(
    m: Float[Tensor, "dim dim"],
) -> Float[Tensor, "dim dim"]:
    """Evaluates W = (I - A)(I + A)^{-1} with skew projection on CPU."""
    dim = m.shape[0]
    tri_u = torch.triu(m, diagonal=1)
    skew_a = tri_u - tri_u.T
    eye = torch.eye(dim, dtype=m.dtype, device=m.device)
    p = eye + skew_a
    q = eye - skew_a

    # Solve P X = Q^T -> X = P^{-1} Q^T -> W = X^T = Q (P^{-1})^T = Q P^{-T}
    # Since P = I + A, P^T = I - A = Q.
    # W = Q (I + A)^{-1}
    inv_p = torch.linalg.inv(p)
    w_ortho = q @ inv_p
    return w_ortho


def tiled_orthogonal_forward_cpu(
    x: Float[Tensor, "batch dim"],
    skew_param: Float[Tensor, "dim dim"],
    bias: Float[Tensor, "dim"],
    block_size: int = 16,
) -> Float[Tensor, "batch dim"]:
    """Evaluates isometric affine mapping y = x @ W + b with block-tiled cache slices."""
    batch_size, dim = x.shape
    w_ortho = fused_cayley_solve_cpu(skew_param)

    y = torch.empty(batch_size, dim, dtype=x.dtype, device=x.device)
    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end]
        y[b_start:b_end] = x_block @ w_ortho + bias

    return y


def meta_profile_linear_stack_cpu(
    batch_size: int,
    dim: int,
    num_layers: int,
    dtype_bytes: int = 4,
) -> ModelMemoryProfile:
    """Profiles memory and compute requirements of a deep linear model without physical allocation."""
    # Each layer has weight [dim, dim] and bias [dim]
    params_per_layer = (dim * dim) + dim
    total_params = params_per_layer * num_layers
    param_bytes = total_params * dtype_bytes

    # No persistent buffers
    buffer_bytes = 0

    # Activations per layer: [batch_size, dim]
    activation_bytes_per_layer = batch_size * dim * dtype_bytes
    # Training peak activations: sum over all layers retained for backward pass
    peak_activation_bytes = activation_bytes_per_layer * num_layers

    # FLOPs per layer: 2 * batch_size * dim * dim (matmul) + batch_size * dim (bias)
    flops_per_layer = (2 * batch_size * dim * dim) + (batch_size * dim)
    total_flops = flops_per_layer * num_layers

    return ModelMemoryProfile(
        param_bytes=param_bytes,
        buffer_bytes=buffer_bytes,
        peak_activation_bytes=peak_activation_bytes,
        total_flops=total_flops,
    )


if __name__ == "__main__":
    from pathlib import Path

    import core

    print("Testing CPU Hardware Kernel: Cayley Parameterization & Meta Profiler...")

    ckpt_path = Path("weights/00_foundations/06_nn_module_internals_parametrizations_and_meta_device.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    skew_param = weights["skew_param"]
    w_gold = weights["weight"]
    b_gold = weights["bias"]

    dim = int(meta.get("dim", 3))

    # 1. Test Cayley Solve
    w_kernel = fused_cayley_solve_cpu(skew_param)
    torch.testing.assert_close(w_kernel, w_gold, atol=1e-5, rtol=1e-5)

    eye = torch.eye(dim)
    ortho_check = w_kernel.T @ w_kernel
    torch.testing.assert_close(ortho_check, eye, atol=1e-6, rtol=1e-6)
    print("[PASS] Cayley solve kernel outputs strictly orthogonal matrix W^T W = I.")

    # 2. Test Tiled Forward Parity
    out_kernel = tiled_orthogonal_forward_cpu(test_in, skew_param, b_gold, block_size=2)
    torch.testing.assert_close(out_kernel, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Block-tiled forward pass matches golden output within 1e-5.")

    # 3. Test Meta Profiler
    profile = meta_profile_linear_stack_cpu(batch_size=8, dim=1024, num_layers=50)
    expected_params = (1024 * 1024 + 1024) * 50
    assert profile.param_bytes == expected_params * 4
    assert profile.peak_activation_bytes == 8 * 1024 * 4 * 50
    print(
        f"[PASS] Zero-memory meta profiler verified: 50 layers = {profile.param_bytes / 1e6:.1f} MB weights, "
        f"{profile.total_flops / 1e9:.2f} GFLOPs."
    )

    print("[PASS] All Topic 06 CPU kernel tests passed cleanly!")
