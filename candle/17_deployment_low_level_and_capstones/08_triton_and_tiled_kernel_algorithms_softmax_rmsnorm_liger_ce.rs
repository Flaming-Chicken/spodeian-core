//! Triton and Tiled Kernel Algorithms Softmax RMSNorm Liger Ce
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Fused RMSNorm: x / sqrt(mean(x^2) + eps) * gamma.
pub fn fused_rmsnorm(x: &Tensor, gamma: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
    let denom = mean_sq.affine(1.0, eps)?.sqrt()?;
    let norm = x.broadcast_div(&denom)?;
    norm.broadcast_mul(gamma)
}

/// Fused SwiGLU: silu(x @ w_gate) * (x @ w_up).
pub fn fused_swiglu(x: &Tensor, w_gate: &Tensor, w_up: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&(x.matmul(w_gate)?))?;
    let up = x.matmul(w_up)?;
    gate.mul(&up)
}

/// Tiled linear + cross-entropy computing loss and input gradients.
pub fn tiled_linear_cross_entropy(
    h: &Tensor,
    w_vocab: &Tensor,
    labels: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, _d) = h.dims2()?;
    let v = w_vocab.dim(1)?;
    let logits = h.matmul(w_vocab)?;
    let log_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
    let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

    let labels_vec = labels.to_vec1::<f32>()?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    let probs_vec = probs.to_vec2::<f32>()?;

    let mut total_loss = 0.0f32;
    let mut d_logits = vec![vec![0.0f32; v]; b];

    for i in 0..b {
        let target = labels_vec[i] as usize;
        total_loss -= log_probs_vec[i][target];
        for j in 0..v {
            d_logits[i][j] = probs_vec[i][j] - if j == target { 1.0 } else { 0.0 };
        }
    }

    let mean_loss = Tensor::new(&[total_loss / (b as f32)], h.device())?;
    let d_logits_tensor = Tensor::new(d_logits, h.device())?;
    let dh = d_logits_tensor
        .matmul(&(w_vocab.t()?))?
        .affine(1.0 / (b as f64), 0.0)?;

    Ok((mean_loss, dh))
}
