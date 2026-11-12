//! Custom Tensor Subclasses Quantized and Tracked Tensors
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Simulates transparent dequantization: W_fp = (W_int - zp) * scale.
pub fn dequantize_affine(
    int_weight: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
) -> Result<Tensor> {
    let w_f32 = int_weight.to_dtype(candle_core::DType::F32)?;
    let centered = w_f32.broadcast_sub(zero_point)?;
    centered.broadcast_mul(scale)
}

/// Forward affine step computing primal output and forward-AD tangent perturbation.
pub fn forward_dual_affine(
    x_primal: &Tensor,
    x_tangent: &Tensor,
    w: &Tensor,
    b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let y_primal = x_primal.matmul(w)?.broadcast_add(b)?;
    let y_tangent = x_tangent.matmul(w)?;
    Ok((y_primal, y_tangent))
}
