//! Visual Autoregressive MaskGIT, VAR, and MAR
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// MaskGIT cosine masking schedule: gamma(r) = cos(pi / 2 * r).
pub fn maskgit_cosine_schedule(r: &Tensor) -> Result<Tensor> {
    let half_pi = 0.5 * std::f64::consts::PI;
    let scaled = r.affine(half_pi, 0.0)?;
    scaled.cos()
}

/// Applies binary mask to replace masked tokens with mask_token vector.
pub fn maskgit_apply_mask(x: &Tensor, mask: &Tensor, mask_token: &Tensor) -> Result<Tensor> {
    let one_minus_mask = mask.affine(-1.0, 1.0)?;
    let term0 = x.broadcast_mul(&one_minus_mask)?;
    let term1 = mask_token.broadcast_mul(mask)?;
    term0 + term1
}

/// VAR coarse-to-fine autoregressive spatial expansion.
pub fn var_coarse_to_fine_upsample(
    coarse_map: &Tensor,
    repeats: usize,
    w_scale: &Tensor,
    b_scale: &Tensor,
) -> Result<Tensor> {
    let upsampled = coarse_map.repeat((1, repeats, 1))?;
    let (b, l_fine, d) = upsampled.dims3()?;
    let up_flat = upsampled.reshape((b * l_fine, d))?;
    let proj = up_flat.matmul(w_scale)?.broadcast_add(b_scale)?;
    let res_flat = proj.gelu_erf()?;
    let residual = res_flat.reshape((b, l_fine, d))?;
    upsampled + residual
}

/// MAR masked diffusion continuous loss over masked positions.
pub fn mar_masked_diffusion_loss(
    v_pred: &Tensor,
    v_target: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let diff = (v_pred - v_target)?;
    let sq_err = diff.sqr()?;
    let masked_sq_err = sq_err.broadcast_mul(mask)?;
    let sum_err = masked_sq_err.sum_all()?.to_scalar::<f32>()?;
    let (_b, _l, d) = v_pred.dims3()?;
    let mask_sum = mask.sum_all()?.to_scalar::<f32>()?;
    let loss_val = sum_err / (mask_sum * (d as f32) + 1e-6);
    Tensor::from_slice(&[loss_val], (1,), v_pred.device())
}
