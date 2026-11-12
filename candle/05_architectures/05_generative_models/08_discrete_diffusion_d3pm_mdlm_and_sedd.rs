//! Discrete Diffusion D3PM, MDLM, and SEDD
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes discrete marginal probabilities: q(x_t | x_0) = x_0 @ Q_bar.
pub fn d3pm_forward_marginal(x_0_probs: &Tensor, q_bar: &Tensor) -> Result<Tensor> {
    let (b, l, k) = x_0_probs.dims3()?;
    let x_flat = x_0_probs.reshape((b * l, k))?;
    let out_flat = x_flat.matmul(q_bar)?;
    out_flat.reshape((b, l, k))
}

/// Continuous-time masked diffusion marginal: alpha_t * x_0 + (1 - alpha_t) * e_mask.
pub fn mdlm_continuous_marginal(
    x_0_probs: &Tensor,
    alpha_t: &Tensor,
    mask_idx: usize,
) -> Result<Tensor> {
    let (b, l, k) = x_0_probs.dims3()?;
    let mut mask_vec = vec![0.0f32; b * l * k];
    for i in 0..(b * l) {
        mask_vec[i * k + mask_idx] = 1.0;
    }
    let e_mask = Tensor::from_vec(mask_vec, (b, l, k), x_0_probs.device())?;
    let one_minus_alpha = alpha_t.affine(-1.0, 1.0)?;
    let term0 = x_0_probs.broadcast_mul(alpha_t)?;
    let term1 = e_mask.broadcast_mul(&one_minus_alpha)?;
    term0 + term1
}

/// Time-conditioned categorical token scoring and denoising MLP.
pub fn discrete_denoiser(
    x_t: &Tensor,
    t: &Tensor,
    w_x: &Tensor,
    b_x: &Tensor,
    w_t: &Tensor,
    b_t: &Tensor,
    w_out: &Tensor,
    b_out: &Tensor,
) -> Result<Tensor> {
    let (b, l, k) = x_t.dims3()?;
    let d_h = b_x.dims1()?;

    let t_proj = t.matmul(w_t)?.broadcast_add(b_t)?;
    let t_emb = candle_nn::ops::silu(&t_proj)?.unsqueeze(1)?;

    let x_flat = x_t.reshape((b * l, k))?;
    let x_proj = x_flat
        .matmul(w_x)?
        .broadcast_add(b_x)?
        .reshape((b, l, d_h))?;

    let h_pre = x_proj.broadcast_add(&t_emb)?;
    let h = candle_nn::ops::silu(&h_pre)?;

    let h_flat = h.reshape((b * l, d_h))?;
    let logits_flat = h_flat.matmul(w_out)?.broadcast_add(b_out)?;
    logits_flat.reshape((b, l, k))
}

/// Discrete score entropy loss between predicted ratio and ground truth.
pub fn sedd_score_entropy(score_ratio: &Tensor, target_ratio: &Tensor) -> Result<Tensor> {
    let diff = (score_ratio - target_ratio)?;
    diff.sqr()?.mean_all()?.reshape((1,))
}
