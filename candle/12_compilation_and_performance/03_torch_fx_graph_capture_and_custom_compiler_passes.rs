//! Torch Fx Graph Capture and Custom Compiler Passes
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Algebraic folding of BatchNorm into Linear parameters:
/// W_fold = W * (gamma / sqrt(var + eps))
/// b_fold = (b - mean) * (gamma / sqrt(var + eps)) + beta
pub fn algebraic_linear_bn_folding(
    w: &Tensor,
    b: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let var_eps = running_var.affine(1.0, eps as f64)?.sqrt()?;
    let scale = (gamma / var_eps)?;
    let scale_2d = scale.unsqueeze(0)?;
    let w_fold = w.broadcast_mul(&scale_2d)?;
    let b_fold = (b.sub(running_mean)?.mul(&scale)?.add(beta))?;
    Ok((w_fold, b_fold))
}

/// Unfolded sequential Linear -> BatchNorm forward pass.
pub fn sequential_linear_bn(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let z = x.matmul(w)?.broadcast_add(b)?;
    let var_eps = running_var.affine(1.0, eps as f64)?.sqrt()?;
    let norm = z.broadcast_sub(running_mean)?.broadcast_div(&var_eps)?;
    norm.broadcast_mul(gamma)?.broadcast_add(beta)
}

/// Horizontal GEMM fusion pass (e.g. Q, K, V projection concatenation).
pub fn parallel_gemm_fusion(
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    w3: &Tensor,
    b3: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let w_fused = Tensor::cat(&[w1, w2, w3], 1)?;
    let b_fused = Tensor::cat(&[b1, b2, b3], 0)?;
    Ok((w_fused, b_fused))
}
