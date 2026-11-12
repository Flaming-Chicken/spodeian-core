//! Torch Dispatch Modes Flop Counter and Memory Tracker
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// 2-layer residual MLP matching PyTorch tracked forward step.
pub fn mlp_forward(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let lin1 = x.matmul(w1)?.broadcast_add(b1)?;
    let z1 = candle_nn::ops::silu(&lin1)?;
    let z2 = z1.matmul(w2)?.broadcast_add(b2)?;
    let y = (z2 + x)?;
    Ok((y, z1))
}

/// Analytical FLOP counter for 2-layer residual MLP.
pub fn compute_mlp_flops(m: usize, k: usize, n: usize) -> usize {
    let mm1 = 2 * m * k * n;
    let bias1 = m * n;
    let act1 = m * n;
    let mm2 = 2 * m * n * n;
    let bias2 = m * n;
    let res_add = m * n;
    mm1 + bias1 + act1 + mm2 + bias2 + res_add
}
