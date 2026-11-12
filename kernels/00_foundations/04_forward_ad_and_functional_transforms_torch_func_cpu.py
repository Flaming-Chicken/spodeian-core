"""Topic 04: Forward-Mode AD & Functional Transforms (Fused CPU Kernel).

Demonstrates a hardware-aligned cache-tiled CPU simulator evaluating:
1. Joint primal forward evaluation and forward-mode directional derivative (JVP: J @ v).
2. Tiled vector accumulation across L1/L2 cache blocks to eliminate redundant DRAM round-trips.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def fused_forward_jvp_cpu(
    x: Float[Tensor, "batch in_dim"],
    tangents: Float[Tensor, "batch in_dim"],
    weight: Float[Tensor, "in_dim out_dim"],
    bias: Float[Tensor, "*out_dim"],
    block_size: int = 16,
) -> tuple[Float[Tensor, "batch out_dim"], Float[Tensor, "batch out_dim"]]:
    """Fused CPU kernel evaluating primal Y and directional tangent dY simultaneously.

    Simulates SRAM/cache block-tiling where X and tangents share the same memory cache lines
    during the matrix multiplication by weight W:
        Y = X @ W + b
        dY = tangents @ W
    """
    batch_size, in_dim = x.shape
    _, out_dim = weight.shape

    y = torch.zeros(batch_size, out_dim, dtype=x.dtype, device=x.device)
    dy = torch.zeros(batch_size, out_dim, dtype=x.dtype, device=x.device)

    # Cache-tiled block loop simulating SRAM tile loading
    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end, :]
        v_block = tangents[b_start:b_end, :]

        # Compute both projections within the same cache pass
        y[b_start:b_end, :] = x_block @ weight + bias
        dy[b_start:b_end, :] = v_block @ weight

    return y, dy


if __name__ == "__main__":
    import core

    print("Testing CPU Fused Kernel for Topic 04 (Forward AD)...")
    ckpt_path = "weights/00_foundations/04_forward_ad_and_functional_transforms_torch_func.safetensors"
    weights, test_in, test_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    w = weights["weight"]
    b = weights["bias"]
    v_in = weights["tangent_in"]
    v_out_golden = weights["tangent_out"]

    y_fused, dy_fused = fused_forward_jvp_cpu(test_in, v_in, w, b)

    torch.testing.assert_close(y_fused, test_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dy_fused, v_out_golden, atol=1e-5, rtol=1e-5)
    print("[PASS] CPU Fused Kernel verified against golden forward & tangent outputs!")
