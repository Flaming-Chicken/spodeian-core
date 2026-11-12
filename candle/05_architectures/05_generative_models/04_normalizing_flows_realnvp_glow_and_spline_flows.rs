//! Normalizing Flows Realnvp Glow and Spline Flows
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// RealNVP affine coupling forward pass:
/// y_a = x_a
/// s = tanh(x_a @ s_w + s_b)
/// t = x_a @ t_w + t_b
/// y_b = x_b * exp(s) + t
/// log_det = sum(s, dim=-1)
pub fn realnvp_affine_coupling_forward(
    x_a: &Tensor,
    x_b: &Tensor,
    s_w: &Tensor,
    s_b: &Tensor,
    t_w: &Tensor,
    t_b: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let s = x_a.matmul(s_w)?.broadcast_add(s_b)?.tanh()?;
    let t = x_a.matmul(t_w)?.broadcast_add(t_b)?;
    let y_a = x_a.clone();
    let y_b = x_b.mul(&s.exp()?)?.add(&t)?;
    let log_det = s.sum_keepdim(1)?.squeeze(1)?;
    Ok((y_a, y_b, log_det))
}

/// RealNVP affine coupling inverse pass:
/// x_a = y_a
/// s = tanh(y_a @ s_w + s_b)
/// t = y_a @ t_w + t_b
/// x_b = (y_b - t) * exp(-s)
/// log_det = -sum(s, dim=-1)
pub fn realnvp_affine_coupling_inverse(
    y_a: &Tensor,
    y_b: &Tensor,
    s_w: &Tensor,
    s_b: &Tensor,
    t_w: &Tensor,
    t_b: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let s = y_a.matmul(s_w)?.broadcast_add(s_b)?.tanh()?;
    let t = y_a.matmul(t_w)?.broadcast_add(t_b)?;
    let x_a = y_a.clone();
    let neg_s = (&s * (-1.0))?;
    let x_b = y_b.sub(&t)?.mul(&neg_s.exp()?)?;
    let log_det = (s.sum_keepdim(1)?.squeeze(1)? * (-1.0))?;
    Ok((x_a, x_b, log_det))
}

/// Glow invertible linear transform forward: y = x @ w.
pub fn glow_forward(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.matmul(w)
}
