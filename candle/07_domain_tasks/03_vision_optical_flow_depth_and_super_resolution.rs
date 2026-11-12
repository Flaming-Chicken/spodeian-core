//! Vision Optical Flow Depth and Super Resolution
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Sub-pixel convolution (PixelShuffle):
/// Rearranges elements in tensor of shape [B, C * r^2, H, W] to [B, C, H * r, W * r].
pub fn pixel_shuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c_r2, h, w) = x.dims4()?;
    let c = c_r2 / (r * r);
    let reshaped = x.reshape((b, c, r, r, h, w))?;
    let perm = reshaped.permute((0, 1, 4, 2, 5, 3))?;
    perm.contiguous()?.reshape((b, c, h * r, w * r))
}

/// Scale-Invariant Logarithmic Depth Loss (Eigen et al.):
/// d = log(pred + eps) - log(target + eps)
/// L_si = mean(d^2) - alpha * (mean(d))^2
pub fn scale_invariant_depth_loss(
    pred_depth: &Tensor,
    target_depth: &Tensor,
    alpha: f64,
    eps: f64,
) -> Result<Tensor> {
    let (_b, h, w) = pred_depth.dims3()?;
    let n = (h * w) as f64;
    let log_pred = (pred_depth + eps)?.log()?;
    let log_target = (target_depth + eps)?.log()?;
    let d = (log_pred - log_target)?;
    let sum_d2 = d.sqr()?.sum(2)?.sum(1)?;
    let sum_d = d.sum(2)?.sum(1)?;
    let mean_d2 = (sum_d2 / n)?;
    let mean_d = (sum_d / n)?;
    let penalty = (mean_d.sqr()? * alpha)?;
    mean_d2.sub(&penalty)
}

/// Charbonnier Loss (smooth L1 / robust photometric loss):
/// L = mean(sqrt((pred - target)^2 + eps^2))
pub fn charbonnier_loss(pred: &Tensor, target: &Tensor, eps: f64) -> Result<Tensor> {
    let (_b, h, w) = pred.dims3()?;
    let n = (h * w) as f64;
    let diff = (pred - target)?;
    let eps_sq = eps * eps;
    let inner = (diff.sqr()? + eps_sq)?;
    let charb = inner.sqrt()?;
    let sum_charb = charb.sum(2)?.sum(1)?;
    sum_charb / n
}

/// RAFT All-Pairs 4D Correlation Volume (computed as spatial dot product):
/// Returns correlation matrix: [B, H*W, H*W] normalized by sqrt(C).
pub fn raft_correlation(fmap1: &Tensor, fmap2: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = fmap1.dims4()?;
    let v1 = fmap1.reshape((b, c, h * w))?;
    let v2 = fmap2.reshape((b, c, h * w))?;
    let v1_t = v1.transpose(1, 2)?;
    let scale = 1.0 / (c as f64).sqrt();
    v1_t.matmul(&v2)? * scale
}
