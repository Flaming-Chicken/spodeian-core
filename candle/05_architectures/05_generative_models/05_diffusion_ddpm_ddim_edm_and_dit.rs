//! Diffusion DDPM Ddim Edm and DiT
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// DDPM forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise.
pub fn ddpm_forward_diffuse(x_0: &Tensor, noise: &Tensor, alpha_bar: &Tensor) -> Result<Tensor> {
    let sqrt_ab = alpha_bar.sqrt()?;
    let sqrt_one_minus = alpha_bar.affine(-1.0, 1.0)?.sqrt()?;
    let term1 = x_0.broadcast_mul(&sqrt_ab)?;
    let term2 = noise.broadcast_mul(&sqrt_one_minus)?;
    term1.add(&term2)
}

/// Deterministic DDIM step (eta = 0).
pub fn ddim_step(
    x_t: &Tensor,
    eps_pred: &Tensor,
    alpha_bar_t: &Tensor,
    alpha_bar_prev: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let sqrt_ab_t = alpha_bar_t.sqrt()?;
    let sqrt_one_minus_t = alpha_bar_t.affine(-1.0, 1.0)?.sqrt()?;
    let diff = x_t.sub(&eps_pred.broadcast_mul(&sqrt_one_minus_t)?)?;
    let x_0_pred = diff.broadcast_div(&sqrt_ab_t)?;

    let sqrt_ab_prev = alpha_bar_prev.sqrt()?;
    let sqrt_one_minus_prev = alpha_bar_prev.affine(-1.0, 1.0)?.sqrt()?;
    let x_prev = x_0_pred
        .broadcast_mul(&sqrt_ab_prev)?
        .add(&eps_pred.broadcast_mul(&sqrt_one_minus_prev)?)?;
    Ok((x_0_pred, x_prev))
}

/// Classifier-Free Guidance (CFG): eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond).
pub fn classifier_free_guidance(
    eps_uncond: &Tensor,
    eps_cond: &Tensor,
    cfg_scale: f64,
) -> Result<Tensor> {
    let diff = eps_cond.sub(eps_uncond)?;
    let scaled = (diff * cfg_scale)?;
    eps_uncond.add(&scaled)
}

/// DiT adaLN-Zero modulation: y = alpha * (LayerNorm(x) * (1 + gamma) + beta).
pub fn adaln_zero_modulation(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    alpha: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let mean = x.mean_keepdim(1)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(1)?;
    let inv_std = (var + eps)?.powf(-0.5)?;
    let norm = diff.broadcast_mul(&inv_std)?;

    let one_plus_gamma = (gamma + 1.0)?;
    let h = norm.mul(&one_plus_gamma)?.add(beta)?;
    alpha.mul(&h)
}
