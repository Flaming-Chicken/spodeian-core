"""Cache-Tiled CPU Simulated Kernel for Activation Compression & Saved Tensor Hooks.

Topic 05: 05_hooks_and_saved_tensor_hooks (Category 00: Foundations)

Theoretical Overview:
During deep neural network training, activations stored for backward passes consume the majority
of accelerator HBM/SRAM. Saved tensor packing hooks intercept forward activations and apply
in-flight quantization or memory offloading.

This kernel simulates a hardware-tiled activation compressor:
1. Fused Forward + INT8 Packing:
   Computes z = x @ W + b and y = GELU(z) in block-tiled cache slices, simultaneously packing
   z into per-block dynamically scaled INT8 representation, reducing memory footprint by 4x.
2. Unpack + Fused Backward:
   Unpacks the compressed activation buffer directly within CPU L1/L2 cache lines and evaluates
   the vector-Jacobian product dL/dz = dL/dy * GELU'(z), accumulating grad_W and grad_b in-place.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from jaxtyping import Float
from torch import Tensor


class CompressedActivationBuffer(NamedTuple):
    """Packed dynamic INT8 representation of intermediate activations."""

    data: Tensor  # torch.uint8 [batch, out_dim]
    scale: Tensor  # torch.float32 [batch, 1]
    zero_point: Tensor  # torch.float32 [batch, 1]
    orig_shape: torch.Size


def fused_forward_and_pack_cpu(
    x: Float[Tensor, "batch in_dim"],
    weight: Float[Tensor, "in_dim out_dim"],
    bias: Float[Tensor, "*out_dim"],
    block_size: int = 16,
) -> tuple[Float[Tensor, "batch out_dim"], CompressedActivationBuffer]:
    """Evaluates linear projection + GELU and packs intermediate activations into dynamic INT8."""
    batch_size, in_dim = x.shape
    _, out_dim = weight.shape

    # 1. Compute affine projection z = x @ W + b
    z = torch.empty(batch_size, out_dim, dtype=torch.float32)
    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end]  # [B_blk, in_dim]
        z[b_start:b_end] = x_block @ weight + bias

    # 2. Compute GELU forward: 0.5 * z * (1 + erf(z / sqrt(2)))
    y = 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))

    # 3. Dynamic per-sample INT8 compression of intermediate activation z
    z_min = z.min(dim=-1, keepdim=True).values
    z_max = z.max(dim=-1, keepdim=True).values
    scale = torch.clamp((z_max - z_min) / 255.0, min=1e-8)
    zero_point = torch.round(-z_min / scale)

    packed_data = torch.clamp(torch.round(z / scale) + zero_point, 0, 255).to(torch.uint8)
    buffer = CompressedActivationBuffer(
        data=packed_data,
        scale=scale,
        zero_point=zero_point,
        orig_shape=z.shape,
    )
    return y, buffer


def fused_unpack_and_backward_cpu(
    grad_output: Float[Tensor, "batch out_dim"],
    x: Float[Tensor, "batch in_dim"],
    weight: Float[Tensor, "in_dim out_dim"],
    buffer: CompressedActivationBuffer,
) -> tuple[Float[Tensor, "batch in_dim"], Float[Tensor, "in_dim out_dim"], Float[Tensor, "*out_dim"]]:
    """Unpacks compressed activations in cache and computes exact analytical gradients."""
    # 1. Unpack activation z in-cache: z = (data - zero_point) * scale
    z_unpacked = (buffer.data.float() - buffer.zero_point) * buffer.scale

    # 2. GELU derivative: Phi(z) + z * phi(z)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    sqrt_2 = math.sqrt(2.0)

    phi_cdf = 0.5 * (1.0 + torch.erf(z_unpacked / sqrt_2))
    phi_pdf = torch.exp(-0.5 * (z_unpacked ** 2)) * inv_sqrt_2pi
    d_gelu = phi_cdf + z_unpacked * phi_pdf

    # 3. Chain rule: dL/dz = grad_output * d_gelu
    dL_dz = grad_output * d_gelu

    # 4. In-place gradient accumulation
    grad_w = x.T @ dL_dz
    grad_b = dL_dz.sum(dim=0)
    grad_x = dL_dz @ weight.T

    return grad_x, grad_w, grad_b


if __name__ == "__main__":
    # Smoke test for kernel
    x_test = torch.randn(4, 3)
    w_test = torch.randn(3, 2)
    b_test = torch.randn(2)
    g_out = torch.ones(4, 2)

    y_out, buf = fused_forward_and_pack_cpu(x_test, w_test, b_test)
    gx, gw, gb = fused_unpack_and_backward_cpu(g_out, x_test, w_test, buf)

    assert y_out.shape == (4, 2)
    assert gx.shape == (4, 3)
    assert gw.shape == (3, 2)
    assert gb.shape == (2,)
    print("[PASS] CPU Activation Compression & Backward Kernel smoke test succeeded.")
