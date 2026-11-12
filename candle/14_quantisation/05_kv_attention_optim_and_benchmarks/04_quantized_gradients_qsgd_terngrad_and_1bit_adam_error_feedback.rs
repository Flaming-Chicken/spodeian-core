//! Quantized Gradients QSGD Terngrad and 1bit Adam Error Feedback
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Quantized SGD (QSGD) with L2 norm scaling and uniform coordinate quantization.
pub fn qsgd_quantize(grad: &Tensor, num_levels: usize) -> Result<(Tensor, Tensor)> {
    let norm = grad.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norm_eps = norm.affine(1.0, 1e-8)?;
    let xi = grad.abs()?.broadcast_div(&norm_eps)?;
    let num_levels_f = num_levels as f64;
    let tilde_xi = (xi * num_levels_f)?
        .round()?
        .affine(1.0 / num_levels_f, 0.0)?;
    let scaled = norm.broadcast_mul(&tilde_xi)?;
    let quant_grad = (grad.sign()? * scaled)?;
    Ok((quant_grad, norm))
}

/// TernGrad 3-level {-scale, 0, +scale} gradient quantization with outlier clipping.
pub fn terngrad_quantize(grad: &Tensor, clip_factor: f64) -> Result<(Tensor, Tensor)> {
    let mean = grad.mean_keepdim(1)?;
    let centered = grad.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let std = var.affine(1.0, 1e-8)?.sqrt()?;
    let c = (std * clip_factor)?;
    let c_pos = c.broadcast_as(grad.shape())?;
    let c_neg = c.neg()?.broadcast_as(grad.shape())?;
    let g_clip = grad.clamp(&c_neg, &c_pos)?;
    let scale = g_clip.abs()?.max_keepdim(1)?;
    let scale_eps = scale.affine(1.0, 1e-8)?;
    let norm_g = g_clip.abs()?.broadcast_div(&scale_eps)?;
    let mask = norm_g.ge(0.5f32)?.to_dtype(grad.dtype())?;
    let quant_grad = (g_clip.sign()? * mask)?.broadcast_mul(&scale)?;
    Ok((quant_grad, scale))
}

/// 1-bit Adam momentum compression with Error Feedback (EF).
pub fn onebit_adam_compress(
    momentum: &Tensor,
    error_buffer: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let m_tilde = (momentum + error_buffer)?;
    let scale = m_tilde.abs()?.mean_keepdim(1)?;
    let mask = m_tilde.ge(0.0f32)?.to_dtype(candle_core::DType::F32)?;
    let signs = mask.affine(2.0, -1.0)?;
    let m_hat = signs.broadcast_mul(&scale)?;
    let new_error = (m_tilde - &m_hat)?;
    Ok((m_hat, new_error, scale))
}

/// Applies a parameter update using 1-bit compressed momentum and local variance.
pub fn onebit_adam_update_step(
    param: &Tensor,
    m_hat: &Tensor,
    v: &Tensor,
    lr: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<Tensor> {
    let denom = v.sqrt()?.affine(1.0, eps)?;
    let scaled_m = m_hat.broadcast_div(&denom)?;
    let decay = (param * weight_decay)?;
    let step = (&scaled_m + &decay)?;
    let delta = (step * lr)?;
    param - delta
}
