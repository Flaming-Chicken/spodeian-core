//! Regularization and Training Stabilizers: Dropout, DropPath, Soft-Capping, Label Smoothing, and Router Z-Loss (C03 Topic 06).
//!
//! Native Rust Candle implementation of:
//! 1. Inverted Dropout with deterministic mask (Srivastava et al. 2014).
//! 2. DropPath / Stochastic Depth (Huang et al. 2016).
//! 3. Logit Soft-Capping (Gemma / Grok / DeepSeek).
//! 4. Label Smoothing Cross-Entropy (Szegedy et al. 2016).
//! 5. Router Z-Loss (ST-MoE, Fedus et al. 2022).
//! 6. Linear Anchor.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// Inverted Dropout forward with deterministic mask: (x * mask) / (1 - p).
pub fn dropout_forward(x: &Tensor, mask: &Tensor, p: f64) -> Result<Tensor> {
    let scale = 1.0 / (1.0 - p);
    let scaled_mask = (mask * scale)?;
    x.mul(&scaled_mask)
}

/// DropPath / Stochastic Depth forward: x_identity + (x_branch * mask) / (1 - drop_prob).
pub fn droppath_forward(
    x_identity: &Tensor,
    x_branch: &Tensor,
    mask: &Tensor,
    drop_prob: f64,
) -> Result<Tensor> {
    let scale = 1.0 / (1.0 - drop_prob);
    let scaled_branch = (x_branch.broadcast_mul(mask)? * scale)?;
    x_identity.add(&scaled_branch)
}

/// Logit Soft-Capping forward: cap * tanh(x / cap).
pub fn softcapping_forward(x: &Tensor, cap: f64) -> Result<Tensor> {
    let scaled = (x / cap)?;
    let tanh_x = scaled.tanh()?;
    tanh_x * cap
}

/// Label Smoothing Cross-Entropy forward: (1 - alpha) * NLL + (alpha / K) * sum(-log_probs).
pub fn label_smoothing_loss_forward(
    logits: &Tensor,
    targets: &Tensor,
    alpha: f64,
) -> Result<Tensor> {
    let (b, k) = logits.dims2()?;
    let max_logits = logits.max_keepdim(1)?;
    let lse = logits
        .broadcast_sub(&max_logits)?
        .exp()?
        .sum_keepdim(1)?
        .log()?
        .broadcast_add(&max_logits)?;
    let log_probs = logits.broadcast_sub(&lse)?;

    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    let targets_vec = targets.flatten_all()?.to_vec1::<f32>()?;

    let mut total_loss = 0.0f32;
    for i in 0..b {
        let target_idx = targets_vec[i] as usize;
        let nll = -log_probs_vec[i][target_idx];
        let mut uniform_loss = 0.0f32;
        for c in 0..k {
            uniform_loss += -log_probs_vec[i][c];
        }
        uniform_loss /= k as f32;
        total_loss += (1.0 - alpha as f32) * nll + (alpha as f32) * uniform_loss;
    }
    let mean_loss = total_loss / (b as f32);
    Tensor::new(&[mean_loss], logits.device())
}

/// Router Z-Loss forward: (z_coef / B) * sum( (log sum exp(z))^2 ).
pub fn router_z_loss_forward(router_logits: &Tensor, z_coef: f64) -> Result<Tensor> {
    let max_logits = router_logits.max_keepdim(1)?;
    let lse = router_logits
        .broadcast_sub(&max_logits)?
        .exp()?
        .sum_keepdim(1)?
        .log()?
        .broadcast_add(&max_logits)?;
    let loss_z = (lse.sqr()?.mean_all()? * z_coef)?;
    loss_z.flatten_all()
}
