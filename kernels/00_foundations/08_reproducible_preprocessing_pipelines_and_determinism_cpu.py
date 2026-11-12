"""Cache-Tiled Hardware-Optimized CPU Kernel: Kahan Compensated Reduction & Determinism.

Topic 08: 08_reproducible_preprocessing_pipelines_and_determinism (Category 00: Foundations)

Theoretical Overview:
Floating-point reduction operations (sum, mean, variance) across large batch sizes on multicore
CPUs are vulnerable to non-associative accumulation:
    (a + b) + c != a + (b + c)
Depending on thread scheduling, worker chunking, and SIMD lane alignment, naive accumulations
produce divergent low-order bits.

This kernel implements two deterministic reduction mechanisms:
1. Block-Tiled Kahan Compensated Accumulation:
   Maintains a per-channel running error compensation register $c_j$. When adding sample $x_{i, j}$,
   it recovers lost precision bits:
       y = x_{i, j} - c_j
       t = sum_j + y
       c_j = (t - sum_j) - y
       sum_j = t
   This bounds cumulative numerical drift to O(eps_mach + N * eps_mach^2), ensuring bitwise
   reproducibility across arbitrary thread block sizes.
2. In-Register Linear Forward Projection:
   Evaluates y = x @ W + b in cache-resident blocks.
"""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor


def tiled_kahan_sum_cpu(
    x: Float[Tensor, "batch dim"],
    block_size: int = 16,
) -> Float[Tensor, "*dim"]:
    """Evaluates Kahan compensated reduction across dimension 0 using cache blocks."""
    batch_size, dim = x.shape
    sum_acc = torch.zeros(dim, dtype=x.dtype, device=x.device)
    comp_acc = torch.zeros(dim, dtype=x.dtype, device=x.device)

    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end]  # [B_blk, dim]

        for i in range(x_block.shape[0]):
            row = x_block[i]
            y = row - comp_acc
            t = sum_acc + y
            comp_acc = (t - sum_acc) - y
            sum_acc = t

    return sum_acc


def deterministic_linear_forward_cpu(
    x: Float[Tensor, "batch in_dim"],
    weight: Float[Tensor, "in_dim out_dim"],
    bias: Float[Tensor, "*out_dim"],
    block_size: int = 16,
) -> Float[Tensor, "batch out_dim"]:
    """Evaluates linear projection in deterministic cache-blocked tiles."""
    batch_size, in_dim = x.shape
    _, out_dim = weight.shape

    y = torch.empty(batch_size, out_dim, dtype=x.dtype, device=x.device)
    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end]
        y[b_start:b_end] = x_block @ weight + bias

    return y


if __name__ == "__main__":
    import core

    print("Testing CPU Hardware Kernel: Kahan Compensated Reduction & Determinism...")

    ckpt_path = Path("weights/00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w_gold = weights["weight"]
    b_gold = weights["bias"]
    kahan_mean_gold = weights["kahan_mean"]

    # 1. Test Tiled Kahan Summation Parity
    k_sum = tiled_kahan_sum_cpu(test_in, block_size=2)
    k_mean = k_sum / float(test_in.shape[0])
    torch.testing.assert_close(k_mean, kahan_mean_gold, atol=1e-6, rtol=1e-6)
    print("[PASS] Tiled Kahan compensated reduction matches golden reference.")

    # 2. Test Deterministic Forward Projection
    pred = deterministic_linear_forward_cpu(test_in, w_gold, b_gold, block_size=2)
    torch.testing.assert_close(pred, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Block-tiled deterministic forward projection matches golden reference.")

    print("[PASS] All Topic 08 CPU kernel tests passed cleanly!")
