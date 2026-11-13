//! Normalization Layers Complete: LayerNorm, RMSNorm, DyT, adaLN-Zero, and Weight Standardization (C03 Topic 02).
//!
//! Native Rust Candle implementation of:
//! 1. LayerNorm: Mean-centering and variance scaling across hidden dimensions (Ba et al. 2016).
//! 2. RMSNorm: Root Mean Square Layer Normalization without mean-centering (Zhang & Sennrich 2019).
//! 3. Dynamic Tanh (DyT): Elementwise reduction-free nonlinear scaling (Zhu et al. 2025).
//! 4. adaLN-Zero: Adaptive LayerNorm with conditioning vector modulation (Peebles & Xie 2023).
//! 5. QK-Norm: Query-Key head-wise RMS normalization (Dehghani et al. 2023).
//! 6. Scaled Weight Standardization: Linear weight standardization across fan-in (Brock et al. 2021).

use candle_core::{D, Result, Tensor};

/// Linear anchor forward.
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

/// LayerNorm: ((x - mean) / sqrt(var + eps)) * gamma + beta
pub fn layernorm_forward(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let norm = centered.broadcast_div(&std)?;
    norm.broadcast_mul(gamma)?.broadcast_add(beta)
}

/// RMSNorm: (x / sqrt(mean(x^2) + eps)) * gamma
pub fn rmsnorm_forward(x: &Tensor, gamma: &Tensor, eps: f64) -> Result<Tensor> {
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    let norm = x.broadcast_div(&rms)?;
    norm.broadcast_mul(gamma)
}

/// Dynamic Tanh (DyT): gamma * tanh(alpha * x) + beta (reduction-free)
pub fn dyt_forward(x: &Tensor, alpha: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let scaled = x.broadcast_mul(alpha)?;
    let tanh_act = scaled.tanh()?;
    tanh_act.broadcast_mul(gamma)?.broadcast_add(beta)
}

/// adaLN-Zero: (1 + gamma_mod) * LayerNorm(x) + beta_mod
pub fn adaln_zero_forward(
    x: &Tensor,
    cond: &Tensor,
    w_adaln: &Tensor,
    b_adaln: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let proj = cond.matmul(w_adaln)?.broadcast_add(b_adaln)?;
    let dim = x.dim(D::Minus1)?;
    let gamma_mod = proj.narrow(D::Minus1, 0, dim)?;
    let beta_mod = proj.narrow(D::Minus1, dim, dim)?;

    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let base_ln = centered.broadcast_div(&std)?;

    let ones = Tensor::ones_like(&gamma_mod)?;
    let scale = (&ones + &gamma_mod)?;
    base_ln.broadcast_mul(&scale)?.broadcast_add(&beta_mod)
}

/// QK-Norm: x / sqrt(mean(x^2) + eps)
pub fn qknorm_forward(x: &Tensor, eps: f64) -> Result<Tensor> {
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)
}

/// Scaled Weight Standardization along fan-in dimension (dim 1)
pub fn weight_standardization_forward(w: &Tensor, eps: f64) -> Result<Tensor> {
    let fan_in = w.dim(1)? as f64;
    let mean = w.mean_keepdim(1)?;
    let centered = w.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let denom = ((var + eps)?.sqrt()? * fan_in.sqrt())?;
    centered.broadcast_div(&denom)
}
