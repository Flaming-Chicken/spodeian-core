//! Mechanistic Interpretability Probing Lens Patching and SAE
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Logit Lens: Project intermediate residual stream hidden states into vocabulary logits.
pub fn logit_lens(
    hidden_states: &Tensor,
    unembed_weight: &Tensor,
    unembed_bias: &Tensor,
) -> Result<Tensor> {
    hidden_states
        .matmul(unembed_weight)?
        .broadcast_add(unembed_bias)
}

/// First-order attribution patching (Syed et al., 2023) approximating activation patching.
pub fn attribution_patching(
    clean_acts: &Tensor,
    corrupt_acts: &Tensor,
    clean_grads: &Tensor,
) -> Result<Tensor> {
    let diff = (corrupt_acts - clean_acts)?;
    let atp = diff.mul(clean_grads)?;
    atp.sum(1)
}

/// Top-K Sparse Autoencoder (Gao et al., 2024).
pub fn topk_sparse_autoencoder(
    x: &Tensor,
    w_enc: &Tensor,
    b_enc: &Tensor,
    w_dec: &Tensor,
    b_dec: &Tensor,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let x_cent = x.broadcast_sub(b_dec)?;
    let pre_act = x_cent.matmul(w_enc)?.broadcast_add(b_enc)?;
    let relu = pre_act.relu()?;

    let n = x.dim(0)?;
    let d_sae = w_enc.dim(1)?;
    let relu_vec = relu.to_vec2::<f32>()?;
    let mut f_z = vec![vec![0.0f32; d_sae]; n];

    for i in 0..n {
        let mut indexed: Vec<(usize, f32)> = relu_vec[i].iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, val) in indexed.iter().take(k) {
            if val > 0.0 {
                f_z[i][idx] = val;
            }
        }
    }

    let f_z_flat: Vec<f32> = f_z.into_iter().flatten().collect();
    let f_z_t = Tensor::from_vec(f_z_flat, (n, d_sae), x.device())?;
    let x_hat = f_z_t.matmul(w_dec)?.broadcast_add(b_dec)?;

    Ok((f_z_t, x_hat))
}

/// JumpReLU hard thresholding without L1 shrinkage bias (Rajamanoharan et al., 2024).
pub fn jumprelu_activation(z: &Tensor, threshold: &Tensor) -> Result<Tensor> {
    let thresh = threshold.to_scalar::<f32>()?;
    let z_vec = z.to_vec1::<f32>()?;
    let out_vec: Vec<f32> = z_vec
        .into_iter()
        .map(|v| if v > thresh { v } else { 0.0 })
        .collect();

    Tensor::from_vec(out_vec, z.shape(), z.device())
}
