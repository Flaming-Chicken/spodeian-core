//! Autoencoders VAE Beta VAE and VQ Fsq Lfq
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// VAE reparameterization trick and analytical KL divergence:
/// z = mu + std * eps
/// kl = -0.5 * beta * sum(1 + logvar - mu^2 - exp(logvar), dim=-1)
pub fn vae_reparameterize_and_kl(
    mu: &Tensor,
    logvar: &Tensor,
    eps: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor)> {
    let half_logvar = (logvar * 0.5)?;
    let std = half_logvar.exp()?;
    let z = (mu + std.mul(eps)?)?;

    let ones = Tensor::ones_like(logvar)?;
    let mu_sq = mu.sqr()?;
    let exp_logvar = logvar.exp()?;

    let inner = ones.add(logvar)?.sub(&mu_sq)?.sub(&exp_logvar)?;
    let sum_kl = inner.sum_keepdim(1)?.squeeze(1)?;
    let kl = (sum_kl * (-0.5 * beta))?;
    Ok((z, kl))
}

/// Finite Scalar Quantization (FSQ):
/// Clamps continuous z to [-half_width, half_width] and rounds to nearest integer.
pub fn finite_scalar_quantization(z: &Tensor, half_width: f64) -> Result<(Tensor, Tensor)> {
    let clamped = z.clamp(-half_width, half_width)?;
    let shape = clamped.shape();
    let vals: Vec<f32> = clamped.flatten_all()?.to_vec1()?;
    let rounded: Vec<f32> = vals.iter().map(|&v| v.round()).collect();
    let indices: Vec<f32> = rounded.iter().map(|&v| v + half_width as f32).collect();

    let dev = z.device();
    let z_q = Tensor::from_vec(rounded, shape, dev)?;
    let fsq_indices = Tensor::from_vec(indices, shape, dev)?;
    Ok((z_q, fsq_indices))
}
