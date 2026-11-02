//! Vision Segmentation Unet Mask2former and SAM Promptable
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Soft Dice Loss:
/// Dice = (2 * sum(p * t) + eps) / (sum(p^2) + sum(t^2) + eps)
/// Loss = 1 - Dice
pub fn dice_loss(pred_mask: &Tensor, target_mask: &Tensor, eps: f64) -> Result<Tensor> {
    let prod = pred_mask.mul(target_mask)?;
    let intersection = prod.sum(2)?.sum(1)?; // [B]

    let p_sq = pred_mask.sqr()?.sum(2)?.sum(1)?;
    let t_sq = target_mask.sqr()?.sum(2)?.sum(1)?;
    let cardinality = (p_sq + t_sq)?;

    let num = ((intersection * 2.0)? + eps)?;
    let den = (cardinality + eps)?;
    let dice = num.div(&den)?;
    let ones = Tensor::ones_like(&dice)?;
    ones.sub(&dice)
}

/// Mask2Former Masked Cross-Attention:
/// Attention logits = Q @ K^T / sqrt(D) + attn_mask
/// where attn_mask = 0 if foreground, -1e4 if background.
pub fn mask2former_masked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: &Tensor,
) -> Result<Tensor> {
    let (_b, _nq, d) = q.dims3()?;
    let scale = 1.0 / (d as f64).sqrt();
    let k_t = k.transpose(1, 2)?;
    let scores = (q.matmul(&k_t)? * scale)?;
    let masked_scores = scores.broadcast_add(attn_mask)?;
    let probs = candle_nn::ops::softmax_last_dim(&masked_scores)?;
    probs.matmul(v)
}

/// SAM (Segment Anything Model) Positional Prompt Encoding via Random Gaussian Fourier Features:
/// gamma(p) = [sin(2 * pi * p @ B^T), cos(2 * pi * p @ B^T)]
pub fn sam_fourier_prompt_embed(points: &Tensor, gaussian_matrix: &Tensor) -> Result<Tensor> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let (b, np, _two) = points.dims3()?;
    let (d_feat, _two2) = gaussian_matrix.dims2()?;
    let pts_2d = points.reshape((b * np, 2))?;
    let b_t = gaussian_matrix.t()?;
    let proj_2d = pts_2d.matmul(&b_t)?;
    let proj = (proj_2d.reshape((b, np, d_feat))? * two_pi)?;
    let sin_p = proj.sin()?;
    let cos_p = proj.cos()?;
    Tensor::cat(&[&sin_p, &cos_p], 2)
}
