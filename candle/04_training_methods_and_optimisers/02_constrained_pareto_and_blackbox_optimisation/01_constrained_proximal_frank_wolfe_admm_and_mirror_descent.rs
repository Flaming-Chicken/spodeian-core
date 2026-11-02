//! Candle Implementation for Category 04 Topic 08 (Subfolder 02 Topic 01):
//! Constrained, Proximal, Frank-Wolfe, ADMM, and Mirror Descent.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Projected Gradient Descent (PGD) with Euclidean ball projection.
//! 2. Frank-Wolfe (Conditional Gradient) on L1 ball.
//! 3. Proximal Gradient (ISTA) with soft-thresholding.
//! 4. Entropic Mirror Descent on probability simplex.

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

/// Projected Gradient Descent with L2 ball projection.
pub fn step_pgd(param: &Tensor, grad: &Tensor, lr: f64, radius: f64, eps: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut v = vec![0.0f64; d];
    let mut sum_sq = 0.0f64;
    for i in 0..d {
        let val = (p_vec[i] as f64) - lr * (g_vec[i] as f64);
        v[i] = val;
        sum_sq += val * val;
    }
    let norm_v = sum_sq.sqrt();
    let scale = if norm_v <= radius {
        1.0
    } else {
        radius / (norm_v + eps)
    };

    let out: Vec<f32> = v.iter().map(|&x| (x * scale) as f32).collect();
    Tensor::from_vec(out, d, param.device())
}

/// Frank-Wolfe step on L1 ball ||s||_1 <= tau.
pub fn step_frank_wolfe(param: &Tensor, grad: &Tensor, tau: f64, gamma: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut max_abs = -1.0f64;
    let mut max_idx = 0;
    for i in 0..d {
        let abs_g = (g_vec[i] as f64).abs();
        if abs_g > max_abs {
            max_abs = abs_g;
            max_idx = i;
        }
    }

    let g_max = g_vec[max_idx] as f64;
    let sign_g = if g_max > 0.0 {
        1.0
    } else if g_max < 0.0 {
        -1.0
    } else {
        0.0
    };

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let s_i = if i == max_idx { -tau * sign_g } else { 0.0 };
        out[i] = ((1.0 - gamma) * (p_vec[i] as f64) + gamma * s_i) as f32;
    }
    Tensor::from_vec(out, d, param.device())
}

/// Proximal Gradient (ISTA) with soft-thresholding.
pub fn step_ista(param: &Tensor, grad: &Tensor, lr: f64, lam: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();
    let thresh = lr * lam;

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let v = (p_vec[i] as f64) - lr * (g_vec[i] as f64);
        let abs_v = v.abs();
        let sign_v = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
        let shrunk = (abs_v - thresh).max(0.0);
        out[i] = (sign_v * shrunk) as f32;
    }
    Tensor::from_vec(out, d, param.device())
}

/// Entropic Mirror Descent on probability simplex.
pub fn step_mirror_descent(param: &Tensor, grad: &Tensor, lr: f64, eps: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut unnorm = vec![0.0f64; d];
    let mut sum_unnorm = 0.0f64;
    for i in 0..d {
        let u = (p_vec[i] as f64) * (-lr * (g_vec[i] as f64)).exp();
        unnorm[i] = u;
        sum_unnorm += u;
    }

    let out: Vec<f32> = unnorm
        .iter()
        .map(|&u| (u / (sum_unnorm + eps)) as f32)
        .collect();
    Tensor::from_vec(out, d, param.device())
}

/// Unified module encapsulating constrained optimization for parity verification.
pub struct ConstrainedOptimUnifiedModule {
    pub param_pgd: Tensor,
    pub grad_pgd: Tensor,
    pub param_fw: Tensor,
    pub grad_fw: Tensor,
    pub param_ista: Tensor,
    pub grad_ista: Tensor,
    pub param_md: Tensor,
    pub grad_md: Tensor,
    pub anchor: LinearAnchor,
    pub golden_param_pgd: Tensor,
    pub golden_param_fw: Tensor,
    pub golden_param_ista: Tensor,
    pub golden_param_md: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl ConstrainedOptimUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let param_pgd = tensors
            .get("param_pgd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_pgd".into()))?;
        let grad_pgd = tensors
            .get("grad_pgd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_pgd".into()))?;
        let param_fw = tensors
            .get("param_fw")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_fw".into()))?;
        let grad_fw = tensors
            .get("grad_fw")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_fw".into()))?;
        let param_ista = tensors
            .get("param_ista")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_ista".into()))?;
        let grad_ista = tensors
            .get("grad_ista")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_ista".into()))?;
        let param_md = tensors
            .get("param_md")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_md".into()))?;
        let grad_md = tensors
            .get("grad_md")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_md".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_param_pgd = tensors
            .get("golden_param_pgd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_pgd".into()))?;
        let golden_param_fw = tensors
            .get("golden_param_fw")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_fw".into()))?;
        let golden_param_ista = tensors
            .get("golden_param_ista")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_ista".into()))?;
        let golden_param_md = tensors
            .get("golden_param_md")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_md".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            param_pgd,
            grad_pgd,
            param_fw,
            grad_fw,
            param_ista,
            grad_ista,
            param_md,
            grad_md,
            anchor,
            golden_param_pgd,
            golden_param_fw,
            golden_param_ista,
            golden_param_md,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: PGD
        let p_pgd = step_pgd(&self.param_pgd, &self.grad_pgd, 0.1, 1.0, 1e-8)?;
        let err_pgd = max_abs_diff(&p_pgd, &self.golden_param_pgd)?;
        if err_pgd >= atol {
            candle_core::bail!("PGD invariant failed: err={:e} >= {:e}", err_pgd, atol);
        }
        errors.insert("pgd_err".to_string(), err_pgd);

        // Invariant 2: Frank-Wolfe
        let p_fw = step_frank_wolfe(&self.param_fw, &self.grad_fw, 1.0, 0.2)?;
        let err_fw = max_abs_diff(&p_fw, &self.golden_param_fw)?;
        if err_fw >= atol {
            candle_core::bail!(
                "Frank-Wolfe invariant failed: err={:e} >= {:e}",
                err_fw,
                atol
            );
        }
        errors.insert("fw_err".to_string(), err_fw);

        // Invariant 3: Proximal ISTA
        let p_ista = step_ista(&self.param_ista, &self.grad_ista, 0.05, 0.1)?;
        let err_ista = max_abs_diff(&p_ista, &self.golden_param_ista)?;
        if err_ista >= atol {
            candle_core::bail!("ISTA invariant failed: err={:e} >= {:e}", err_ista, atol);
        }
        errors.insert("ista_err".to_string(), err_ista);

        // Invariant 4: Entropic Mirror Descent
        let p_md = step_mirror_descent(&self.param_md, &self.grad_md, 0.1, 1e-12)?;
        let err_md = max_abs_diff(&p_md, &self.golden_param_md)?;
        if err_md >= atol {
            candle_core::bail!(
                "Mirror Descent invariant failed: err={:e} >= {:e}",
                err_md,
                atol
            );
        }
        errors.insert("mirror_descent_err".to_string(), err_md);

        // Invariant 5: Linear anchor
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = max_abs_diff(&anchor_out, &self.golden_anchor_out)?;
        if err_anchor >= atol {
            candle_core::bail!(
                "Anchor invariant failed: err={:e} >= {:e}",
                err_anchor,
                atol
            );
        }
        errors.insert("anchor_err".to_string(), err_anchor);

        Ok(errors)
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = a.sub(b)?.abs()?;
    let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(max as f64)
}
