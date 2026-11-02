//! Losses: Regression, Classification, and Barron Adaptive Robust (C03 Topic 07).
//!
//! Native Rust Candle implementation of:
//! 1. Barron's General Adaptive Robust Loss (Barron 2019).
//! 2. Huber Loss (Smooth L1 piecewise regression).
//! 3. Focal Loss (Lin et al. 2017).
//! 4. PolyLoss (Leng et al. 2022).
//! 5. Asymmetric Loss (ASL, Ridnik et al. 2021).
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

/// Barron's General Adaptive Robust Loss forward.
/// rho(r, alpha, c) = (|alpha - 2| / alpha) * ( (( (r/c)^2 / |alpha - 2| + 1 )^(alpha/2)) - 1 )
pub fn barron_robust_loss_forward(r: &Tensor, alpha: f64, c: f64) -> Result<Tensor> {
    let d_alpha = (alpha - 2.0).abs();
    let scaled = (r / c)?;
    let scaled_sq = scaled.sqr()?;
    let inner = ((scaled_sq / d_alpha)? + 1.0)?;
    let pow_inner = inner.powf(alpha / 2.0)?;
    let elem = ((pow_inner - 1.0)? * (d_alpha / alpha))?;
    elem.mean_all()?.flatten_all()
}

/// Huber Loss forward (Smooth L1 piecewise regression with delta).
pub fn huber_loss_forward(r: &Tensor, delta: f64) -> Result<Tensor> {
    let r_vec = r.flatten_all()?.to_vec1::<f32>()?;
    let delta_f = delta as f32;
    let mut total = 0.0f32;
    for &val in &r_vec {
        let abs_v = val.abs();
        if abs_v <= delta_f {
            total += 0.5 * val.powi(2);
        } else {
            total += delta_f * (abs_v - 0.5 * delta_f);
        }
    }
    let mean = total / (r_vec.len() as f32);
    Tensor::new(&[mean], r.device())
}

/// Multi-Class Focal Loss forward: -(1 - p_t)^gamma * log(p_t).
pub fn focal_loss_forward(logits: &Tensor, targets: &Tensor, gamma: f64) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
    let probs = candle_nn::ops::softmax(logits, 1)?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    let probs_vec = probs.to_vec2::<f32>()?;
    let targets_vec = targets.flatten_all()?.to_vec1::<f32>()?;

    let b = targets_vec.len();
    let gamma_f = gamma as f32;
    let mut total = 0.0f32;
    for i in 0..b {
        let t = targets_vec[i] as usize;
        let p_t = probs_vec[i][t];
        let log_p_t = log_probs_vec[i][t];
        total += -(1.0 - p_t).powf(gamma_f) * log_p_t;
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], logits.device())
}

/// Poly-1 Loss forward: -log(p_t) + epsilon_1 * (1 - p_t).
pub fn polyloss_forward(logits: &Tensor, targets: &Tensor, epsilon_1: f64) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
    let probs = candle_nn::ops::softmax(logits, 1)?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    let probs_vec = probs.to_vec2::<f32>()?;
    let targets_vec = targets.flatten_all()?.to_vec1::<f32>()?;

    let b = targets_vec.len();
    let eps_1 = epsilon_1 as f32;
    let mut total = 0.0f32;
    for i in 0..b {
        let t = targets_vec[i] as usize;
        let p_t = probs_vec[i][t];
        let log_p_t = log_probs_vec[i][t];
        total += -log_p_t + eps_1 * (1.0 - p_t);
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], logits.device())
}

/// Asymmetric Loss (ASL) forward for multi-label classification.
pub fn asymmetric_loss_forward(
    logits: &Tensor,
    targets: &Tensor,
    gamma_pos: f64,
    gamma_neg: f64,
    clip_margin: f64,
    eps: f64,
) -> Result<Tensor> {
    let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
    let targets_vec = targets.flatten_all()?.to_vec1::<f32>()?;
    let n = logits_vec.len();

    let gp = gamma_pos as f32;
    let gn = gamma_neg as f32;
    let m = clip_margin as f32;
    let eps_f = eps as f32;

    let mut total = 0.0f32;
    for i in 0..n {
        let z = logits_vec[i];
        let y = targets_vec[i];
        let p = 1.0 / (1.0 + (-z).exp());
        let p_m = (p - m).max(0.0);

        let pos_loss = -y * (1.0 - p).powf(gp) * p.max(eps_f).ln();
        let neg_loss = -(1.0 - y) * p_m.powf(gn) * (1.0 - p_m).max(eps_f).ln();
        total += pos_loss + neg_loss;
    }
    let mean = total / (n as f32);
    Tensor::new(&[mean], logits.device())
}
