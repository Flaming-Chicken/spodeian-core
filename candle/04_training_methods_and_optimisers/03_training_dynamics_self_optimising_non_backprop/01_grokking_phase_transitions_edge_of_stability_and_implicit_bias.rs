//! Candle Implementation for Category 04 Topic 15 (Subfolder 03 Topic 01):
//! Grokking, Phase Transitions, Edge of Stability, and Implicit Bias.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Grokfast EMA gradient filtering for accelerating phase transition / delayed generalization.
//! 2. Edge of Stability (EOS) cubic oscillator and self-stabilizing dynamics.
//! 3. Implicit Bias logistic margin step on linearly separable data.
//! 4. Linear anchor evaluation for transitive multi-track parity.

use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Linear Anchor module for cross-track parity verification.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xw = x.matmul(&self.weight)?;
        xw.broadcast_add(&self.bias)
    }
}

/// Grokfast EMA gradient filtering step (Lee et al., 2024).
///
/// Computes:
/// mu_next = alpha * mu_prev + (1 - alpha) * g_raw
/// g_filtered = g_raw + lambda_gf * mu_next
pub fn step_grokfast(
    g_raw: &Tensor,
    mu_prev: &Tensor,
    alpha: f64,
    lambda_gf: f64,
) -> Result<(Tensor, Tensor)> {
    let g_v = g_raw.to_vec1::<f32>()?;
    let mu_v = mu_prev.to_vec1::<f32>()?;
    let n = g_v.len();

    let a = alpha as f32;
    let lam = lambda_gf as f32;

    let mut mu_next = vec![0.0f32; n];
    let mut g_filtered = vec![0.0f32; n];

    for i in 0..n {
        let m = a * mu_v[i] + (1.0 - a) * g_v[i];
        mu_next[i] = m;
        g_filtered[i] = g_v[i] + lam * m;
    }

    let dev = g_raw.device();
    Ok((
        Tensor::from_vec(g_filtered, n, dev)?,
        Tensor::from_vec(mu_next, n, dev)?,
    ))
}

/// Edge of Stability (EOS) cubic oscillator step (Cohen et al., 2021).
///
/// Computes:
/// grad = H * x + gamma * x^3
/// x_next = x - eta * grad
pub fn step_edge_of_stability(x: &Tensor, h: &Tensor, gamma: &Tensor, eta: f64) -> Result<Tensor> {
    let x_v = x.to_vec1::<f32>()?;
    let h_v = h.to_vec1::<f32>()?;
    let gam_v = gamma.to_vec1::<f32>()?;
    let n = x_v.len();

    let e = eta as f32;
    let mut x_next = vec![0.0f32; n];

    for i in 0..n {
        let xi = x_v[i];
        let grad = h_v[i] * xi + gam_v[i] * (xi * xi * xi);
        x_next[i] = xi - e * grad;
    }

    Tensor::from_vec(x_next, n, x.device())
}

/// Implicit bias gradient step on separable linear classification (Soudry et al., 2018).
///
/// Computes:
/// margin_i = y_i * (x_i dot w)
/// sigmoid_i = 1 / (1 + exp(margin_i))
/// grad = - 1/N sum_i y_i * x_i * sigmoid_i
/// w_next = w - eta * grad
pub fn step_implicit_bias(w: &Tensor, x: &Tensor, y: &Tensor, eta: f64) -> Result<Tensor> {
    let w_v = w.to_vec1::<f32>()?;
    let y_v = y.to_vec1::<f32>()?;
    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;

    let num_samples = y_v.len();
    let num_features = w_v.len();
    let e = eta as f32;

    let mut grad = vec![0.0f32; num_features];

    for i in 0..num_samples {
        let mut dot = 0.0f32;
        let offset = i * num_features;
        for j in 0..num_features {
            dot += x_flat[offset + j] * w_v[j];
        }

        let margin = y_v[i] * dot;
        let sig = 1.0f32 / (1.0f32 + margin.exp());
        let scalar = -y_v[i] * sig;

        for j in 0..num_features {
            grad[j] += scalar * x_flat[offset + j];
        }
    }

    let inv_n = 1.0f32 / (num_samples as f32);
    let mut w_next = vec![0.0f32; num_features];
    for j in 0..num_features {
        grad[j] *= inv_n;
        w_next[j] = w_v[j] - e * grad[j];
    }

    Tensor::from_vec(w_next, num_features, w.device())
}

/// Unified module encapsulating Grokfast, EOS, and Implicit Bias algorithms.
pub struct TrainingDynamicsUnifiedModule {
    pub g_raw: Tensor,
    pub mu_prev: Tensor,
    pub golden_mu: Tensor,
    pub golden_g_grokfast: Tensor,

    pub x_eos: Tensor,
    pub h_eos: Tensor,
    pub gamma_eos: Tensor,
    pub golden_x_eos: Tensor,

    pub w_sep: Tensor,
    pub x_sep: Tensor,
    pub y_sep: Tensor,
    pub golden_w_sep: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl TrainingDynamicsUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let g_raw = tensors.get("g_raw").unwrap().clone();
        let mu_prev = tensors.get("mu_prev").unwrap().clone();
        let golden_mu = tensors.get("golden_mu").unwrap().clone();
        let golden_g_grokfast = tensors.get("golden_g_grokfast").unwrap().clone();

        let x_eos = tensors.get("x_eos").unwrap().clone();
        let h_eos = tensors.get("h_eos").unwrap().clone();
        let gamma_eos = tensors.get("gamma_eos").unwrap().clone();
        let golden_x_eos = tensors.get("golden_x_eos").unwrap().clone();

        let w_sep = tensors.get("w_sep").unwrap().clone();
        let x_sep = tensors.get("x_sep").unwrap().clone();
        let y_sep = tensors.get("y_sep").unwrap().clone();
        let golden_w_sep = tensors.get("golden_w_sep").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            g_raw,
            mu_prev,
            golden_mu,
            golden_g_grokfast,
            x_eos,
            h_eos,
            gamma_eos,
            golden_x_eos,
            w_sep,
            x_sep,
            y_sep,
            golden_w_sep,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. Grokfast invariant
        let (g_gf, mu_next) = step_grokfast(&self.g_raw, &self.mu_prev, 0.8, 2.0)?;
        let err_mu = (&mu_next - &self.golden_mu)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_gf = (&g_gf - &self.golden_g_grokfast)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_grok = err_mu.max(err_gf);
        assert!(
            err_grok < atol,
            "Grokfast invariant failed: {err_grok:.2e} >= {atol:.2e}"
        );
        errs.insert("grokfast_err".to_string(), err_grok);

        // 2. EOS invariant
        let x_next = step_edge_of_stability(&self.x_eos, &self.h_eos, &self.gamma_eos, 0.1)?;
        let err_eos = (&x_next - &self.golden_x_eos)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_eos < atol,
            "EOS invariant failed: {err_eos:.2e} >= {atol:.2e}"
        );
        errs.insert("eos_err".to_string(), err_eos);

        // 3. Implicit bias invariant
        let w_next = step_implicit_bias(&self.w_sep, &self.x_sep, &self.y_sep, 0.2)?;
        let err_bias = (&w_next - &self.golden_w_sep)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_bias < atol,
            "Implicit bias invariant failed: {err_bias:.2e} >= {atol:.2e}"
        );
        errs.insert("implicit_bias_err".to_string(), err_bias);

        // 4. Linear anchor invariant
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = (&anchor_out - &self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_anchor < atol,
            "Anchor invariant failed: {err_anchor:.2e} >= {atol:.2e}"
        );
        errs.insert("anchor_err".to_string(), err_anchor);

        Ok(errs)
    }
}
