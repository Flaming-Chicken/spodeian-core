//! Time Series Patchtst Nbeats Chronos and Probabilistic
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Reversible Instance Normalization (RevIN) forward normalization.
pub fn revin_normalize(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mean = x.mean_keepdim(2)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(2)?;
    let stdev = (var + eps)?.sqrt()?;
    let x_norm = diff
        .broadcast_div(&stdev)?
        .broadcast_mul(gamma)?
        .broadcast_add(beta)?;
    Ok((x_norm, mean, stdev))
}

/// RevIN reverse denormalization on forecasted horizon.
pub fn revin_denormalize(
    y: &Tensor,
    mean: &Tensor,
    stdev: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    y.broadcast_sub(beta)?
        .broadcast_div(gamma)?
        .broadcast_mul(stdev)?
        .broadcast_add(mean)
}

/// Extracts overlapping 1D patches from lookback time series.
/// x: [B, M, L]
/// Returns: [B * M, N_patches, patch_len]
pub fn patchtst_patching(x: &Tensor, patch_len: usize, stride: usize) -> Result<Tensor> {
    let (b, m, l) = x.dims3()?;
    let n_patches = (l - patch_len) / stride + 1;
    let mut patches_vec = Vec::with_capacity(n_patches);
    for i in 0..n_patches {
        let slice = x.narrow(2, i * stride, patch_len)?.unsqueeze(2)?;
        patches_vec.push(slice);
    }
    let slice_refs: Vec<&Tensor> = patches_vec.iter().collect();
    let stacked = Tensor::cat(&slice_refs, 2)?;
    stacked.reshape((b * m, n_patches, patch_len))
}

/// N-BEATS generic doubly-residual block in Candle.
pub fn nbeats_block(
    x: &Tensor,
    w_fc1: &Tensor,
    b_fc1: &Tensor,
    w_fc2: &Tensor,
    b_fc2: &Tensor,
    w_theta_b: &Tensor,
    w_theta_f: &Tensor,
    basis_b: &Tensor,
    basis_f: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let h1 = x.matmul(w_fc1)?.broadcast_add(b_fc1)?.relu()?;
    let h2 = h1.matmul(w_fc2)?.broadcast_add(b_fc2)?.relu()?;
    let theta_b = h2.matmul(w_theta_b)?;
    let theta_f = h2.matmul(w_theta_f)?;
    let backcast = theta_b.matmul(basis_b)?;
    let forecast = theta_f.matmul(basis_f)?;
    Ok((backcast, forecast))
}

/// Computes pinball (quantile) loss across multiple quantiles.
pub fn pinball_quantile_loss(
    y_true: &Tensor,
    y_pred_quantiles: &Tensor,
    quantiles: &Tensor,
) -> Result<Tensor> {
    let y_true_exp = y_true.unsqueeze(1)?;
    let err = y_true_exp.broadcast_sub(y_pred_quantiles)?;
    let q = quantiles.unsqueeze(0)?.unsqueeze(2)?;
    let q_err = err.broadcast_mul(&q)?;
    let q_minus_1 = (q - 1.0)?;
    let q_minus_1_err = err.broadcast_mul(&q_minus_1)?;
    let loss = q_err.maximum(&q_minus_1_err)?;
    loss.mean_all()
}
