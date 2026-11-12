//! Custom Cpp and CPU SIMD Extension Mechanics
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Fused SIMD FMA followed by LeakyReLU activation.
pub fn fused_simd_fma_leaky_relu(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    negative_slope: f64,
) -> Result<Tensor> {
    let z = ((x * w)? + b)?;
    candle_nn::ops::leaky_relu(&z, negative_slope)
}
