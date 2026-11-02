//! Losses: Proper Scoring Rules, Evidential Deep Learning, and Differentiable Ranking (C03 Topic 11).
//!
//! Native Rust Candle implementation of:
//! 1. Continuous Ranked Probability Score (CRPS) for Gaussian distributions.
//! 2. Brier Score for categorical probabilistic forecasts.
//! 3. Beta-NLL variance-discounted heteroscedastic loss.
//! 4. Evidential Deep Learning Expected MSE under Dirichlet distribution.
//! 5. ListMLE Ranking Loss.
//! 6. Soft-Spearman Differentiable Ranks.
//! 7. Linear Anchor.

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

/// High-precision Abramowitz & Stegun erf approximation (error < 2e-7).
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x_abs);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    sign * (1.0 - poly * (-x_abs * x_abs).exp())
}

/// Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
pub fn crps_gaussian_forward(y: &Tensor, mu: &Tensor, sigma: &Tensor, eps: f32) -> Result<Tensor> {
    let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
    let mu_vec = mu.flatten_all()?.to_vec1::<f32>()?;
    let sig_vec = sigma.flatten_all()?.to_vec1::<f32>()?;
    let n = y_vec.len();

    let sqrt_2 = std::f32::consts::SQRT_2;
    let inv_sqrt_pi = 1.0f32 / std::f32::consts::PI.sqrt();
    let inv_sqrt_2pi = 1.0f32 / (2.0f32 * std::f32::consts::PI).sqrt();

    let mut total = 0.0f32;
    for i in 0..n {
        let s = sig_vec[i].max(eps);
        let z = (y_vec[i] - mu_vec[i]) / s;
        let phi = (-0.5f32 * z * z).exp() * inv_sqrt_2pi;
        let big_phi = 0.5f32 * (1.0f32 + erf_approx(z / sqrt_2));
        let term = s * (z * (2.0f32 * big_phi - 1.0f32) + 2.0f32 * phi - inv_sqrt_pi);
        total += term;
    }
    let mean = total / (n as f32);
    Tensor::new(&[mean], y.device())
}

/// Brier Score: mean squared error across categorical forecast probabilities.
pub fn brier_score_forward(p: &Tensor, y_onehot: &Tensor) -> Result<Tensor> {
    let p_vec = p.to_vec2::<f32>()?;
    let y_vec = y_onehot.to_vec2::<f32>()?;
    let b = p_vec.len();
    let k = p_vec[0].len();

    let mut total = 0.0f32;
    for i in 0..b {
        let mut row_sum = 0.0f32;
        for j in 0..k {
            let diff = p_vec[i][j] - y_vec[i][j];
            row_sum += diff * diff;
        }
        total += row_sum;
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], p.device())
}

/// Beta-NLL: variance-discounted heteroscedastic loss.
pub fn beta_nll_forward(
    y: &Tensor,
    mu: &Tensor,
    var: &Tensor,
    beta: f32,
    eps: f32,
) -> Result<Tensor> {
    let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
    let mu_vec = mu.flatten_all()?.to_vec1::<f32>()?;
    let v_vec = var.flatten_all()?.to_vec1::<f32>()?;
    let n = y_vec.len();

    let mut total = 0.0f32;
    for i in 0..n {
        let v = v_vec[i].max(eps);
        let diff = y_vec[i] - mu_vec[i];
        let loss = 0.5f32 * (v.ln() + (diff * diff) / v);
        let weight = v.powf(beta);
        total += weight * loss;
    }
    let mean = total / (n as f32);
    Tensor::new(&[mean], y.device())
}

/// Evidential Deep Learning (EDL MSE) under Dirichlet distribution.
pub fn edl_mse_forward(alpha: &Tensor, y_onehot: &Tensor) -> Result<Tensor> {
    let a_vec = alpha.to_vec2::<f32>()?;
    let y_vec = y_onehot.to_vec2::<f32>()?;
    let b = a_vec.len();
    let k = a_vec[0].len();

    let mut total = 0.0f32;
    for i in 0..b {
        let mut s = 0.0f32;
        for j in 0..k {
            s += a_vec[i][j];
        }
        let s_sq_s1 = s * s * (s + 1.0f32);
        let mut row_loss = 0.0f32;
        for j in 0..k {
            let a_val = a_vec[i][j];
            let p_hat = a_val / s;
            let diff = y_vec[i][j] - p_hat;
            let var_p = (a_val * (s - a_val)) / s_sq_s1;
            row_loss += diff * diff + var_p;
        }
        total += row_loss;
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], alpha.device())
}

/// ListMLE Ranking Loss.
pub fn listmle_forward(scores: &Tensor, perm: &Tensor) -> Result<Tensor> {
    let s_vec = scores.to_vec2::<f32>()?;
    let p_vec = perm.to_vec2::<i64>()?;
    let b = s_vec.len();
    let k = s_vec[0].len();

    let mut total_loss = 0.0f32;
    for b_idx in 0..b {
        let mut s_ordered = vec![0.0f32; k];
        for i in 0..k {
            let idx = p_vec[b_idx][i] as usize;
            s_ordered[i] = s_vec[b_idx][idx];
        }

        for i in 0..k {
            let mut max_s = f32::NEG_INFINITY;
            for j in i..k {
                if s_ordered[j] > max_s {
                    max_s = s_ordered[j];
                }
            }
            let mut sum_exp = 0.0f32;
            for j in i..k {
                sum_exp += (s_ordered[j] - max_s).exp();
            }
            let log_denom = max_s + sum_exp.ln();
            total_loss -= s_ordered[i] - log_denom;
        }
    }
    let mean = total_loss / ((b * k) as f32);
    Tensor::new(&[mean], scores.device())
}

/// Soft-Spearman Differentiable Ranks.
pub fn soft_ranks_forward(scores: &Tensor, tau: f32) -> Result<Tensor> {
    let s_vec = scores.to_vec2::<f32>()?;
    let b = s_vec.len();
    let k = s_vec[0].len();

    let mut out = vec![vec![0.0f32; k]; b];
    for b_idx in 0..b {
        for i in 0..k {
            let mut r = 0.5f32;
            for j in 0..k {
                let diff = (s_vec[b_idx][i] - s_vec[b_idx][j]) / tau;
                let sig = 1.0f32 / (1.0f32 + (-diff).exp());
                r += sig;
            }
            out[b_idx][i] = r;
        }
    }
    Tensor::new(out, scores.device())
}
