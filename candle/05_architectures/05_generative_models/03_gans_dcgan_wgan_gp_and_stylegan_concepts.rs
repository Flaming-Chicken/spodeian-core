//! GANs Dcgan Wgan GP and Stylegan Concepts
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Adaptive Instance Normalization (AdaIN): y = scale * (x - mean) / sqrt(var + eps) + bias.
pub fn adain(x: &Tensor, scale: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(1)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(1)?;
    let inv_std = (var + eps)?.powf(-0.5)?;
    let norm = diff.broadcast_mul(&inv_std)?;
    norm.mul(scale)?.add(bias)
}

/// StyleGAN2 Weight Modulation and Demodulation.
pub fn stylegan2_weight_demodulation(
    x: &Tensor,
    weights: &Tensor,
    styles: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let w_exp = weights.unsqueeze(0)?; // [1, C_out, C_in]
    let s_exp = styles.unsqueeze(1)?; // [B, 1, C_in]
    let w_mod = w_exp.broadcast_mul(&s_exp)?; // [B, C_out, C_in]

    let w_sq = w_mod.sqr()?;
    let sum_sq = w_sq.sum_keepdim(2)?; // [B, C_out, 1]
    let demod_factors = (sum_sq + eps)?.powf(-0.5)?; // [B, C_out, 1]
    let w_demod = w_mod.broadcast_mul(&demod_factors)?; // [B, C_out, C_in]

    let x_col = x.unsqueeze(2)?; // [B, C_in, 1]
    let y = w_demod.matmul(&x_col)?; // [B, C_out, 1]
    y.squeeze(2)
}

/// WGAN Earth Mover's critic loss: E[D(x_fake)] - E[D(x_real)].
pub fn wgan_critic_loss(real_scores: &Tensor, fake_scores: &Tensor) -> Result<Tensor> {
    let mean_fake = fake_scores.mean_all()?;
    let mean_real = real_scores.mean_all()?;
    mean_fake.sub(&mean_real)
}
