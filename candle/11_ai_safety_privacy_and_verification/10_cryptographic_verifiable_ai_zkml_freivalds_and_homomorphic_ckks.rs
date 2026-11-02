//! Cryptographic Verifiable AI ZKML Freivalds and Homomorphic CKKS
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Freivalds' randomized matrix multiplication check: v = A(Br) - Cr in O(n^2).
pub fn freivalds_verify_residual(a: &Tensor, b: &Tensor, c: &Tensor, r: &Tensor) -> Result<Tensor> {
    let r_col = r.unsqueeze(1)?;
    let br = b.matmul(&r_col)?;
    let abr = a.matmul(&br)?;
    let cr = c.matmul(&r_col)?;
    abr.sub(&cr)?.squeeze(1)
}

/// CKKS / CryptoNets degree-2 polynomial activation: y = a * x^2 + b * x + c.
pub fn ckks_degree2_activation(x: &Tensor, a: f32, b: f32, c: f32) -> Result<Tensor> {
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let out: Vec<f32> = x_vec.into_iter().map(|v| a * v * v + b * v + c).collect();
    Tensor::new(out, x.device())?.reshape(x.shape())
}

/// CKKS degree-4 polynomial activation: y = c4 * x^4 + c2 * x^2 + c0.
pub fn ckks_degree4_activation(x: &Tensor, c4: f32, c2: f32, c0: f32) -> Result<Tensor> {
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let out: Vec<f32> = x_vec
        .into_iter()
        .map(|v| {
            let v2 = v * v;
            c4 * v2 * v2 + c2 * v2 + c0
        })
        .collect();
    Tensor::new(out, x.device())?.reshape(x.shape())
}

/// ZKML fixed-point quantized linear layer: y = round((x @ W + b) * scale) / scale.
pub fn zkml_fixed_point_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let y = x.matmul(weight)?.broadcast_add(bias)?;
    let y_vec = y.to_vec2::<f32>()?;
    let b = y_vec.len();
    let d = y_vec[0].len();
    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            out[i][j] = (y_vec[i][j] * scale).round() / scale;
        }
    }
    Tensor::new(out, x.device())
}
