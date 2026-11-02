//! Candle Implementation for Category 04 Topic 13 (Subfolder 02 Topic 06):
//! Sharpness-Aware Minimization (SAM), SWAD, and Weight Averaging Dynamics.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Sharpness-Aware Minimization (SAM) dual ascent gradient perturbation and update.
//! 2. Lookahead optimizer synchronization between slow and fast weights.
//! 3. Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).
//! 4. Uniform Model Soups weight-space averaging.

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

/// Sharpness-Aware Minimization (SAM) update step.
pub fn step_sam(
    param: &Tensor,
    grad_at_w: &Tensor,
    grad_at_pert: &Tensor,
    rho: f64,
    lr: f64,
    weight_decay: f64,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let p_vec = param.to_vec1::<f32>()?;
    let gw_vec = grad_at_w.to_vec1::<f32>()?;
    let gp_vec = grad_at_pert.to_vec1::<f32>()?;
    let n = p_vec.len();

    let mut sum_sq = 0.0f64;
    for &g in &gw_vec {
        sum_sq += (g as f64) * (g as f64);
    }
    let grad_norm = sum_sq.sqrt() + eps;
    let scale = rho / grad_norm;

    let mut eps_vec = vec![0.0f32; n];
    let mut next_p_vec = vec![0.0f32; n];

    for i in 0..n {
        let e = (gw_vec[i] as f64 * scale) as f32;
        eps_vec[i] = e;
        let update = gp_vec[i] as f64 + weight_decay * (p_vec[i] as f64);
        let next_p = (p_vec[i] as f64) - lr * update;
        next_p_vec[i] = next_p as f32;
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(next_p_vec, n, dev)?,
        Tensor::from_vec(eps_vec, n, dev)?,
    ))
}

/// Lookahead synchronization step.
pub fn step_lookahead(
    slow_param: &Tensor,
    fast_param: &Tensor,
    alpha: f64,
) -> Result<(Tensor, Tensor)> {
    let s_vec = slow_param.to_vec1::<f32>()?;
    let f_vec = fast_param.to_vec1::<f32>()?;
    let n = s_vec.len();

    let mut slow_next = vec![0.0f32; n];
    for i in 0..n {
        let s = s_vec[i] as f64;
        let f = f_vec[i] as f64;
        slow_next[i] = (s + alpha * (f - s)) as f32;
    }
    let fast_next = slow_next.clone();

    let dev = slow_param.device();
    Ok((
        Tensor::from_vec(slow_next, n, dev)?,
        Tensor::from_vec(fast_next, n, dev)?,
    ))
}

/// SWA running average and EMA step.
pub fn step_swa_ema(
    swa_avg: &Tensor,
    ema_avg: &Tensor,
    curr_param: &Tensor,
    n_models: usize,
    decay: f64,
) -> Result<(Tensor, Tensor)> {
    let swa_vec = swa_avg.to_vec1::<f32>()?;
    let ema_vec = ema_avg.to_vec1::<f32>()?;
    let c_vec = curr_param.to_vec1::<f32>()?;
    let n = swa_vec.len();

    let mut next_swa = vec![0.0f32; n];
    let mut next_ema = vec![0.0f32; n];

    let n_m = n_models as f64;
    for i in 0..n {
        let s = swa_vec[i] as f64;
        let e = ema_vec[i] as f64;
        let c = c_vec[i] as f64;

        next_swa[i] = ((s * n_m + c) / (n_m + 1.0)) as f32;
        next_ema[i] = (decay * e + (1.0 - decay) * c) as f32;
    }

    let dev = swa_avg.device();
    Ok((
        Tensor::from_vec(next_swa, n, dev)?,
        Tensor::from_vec(next_ema, n, dev)?,
    ))
}

/// Average a stack of M models: models shape [M, d].
pub fn uniform_model_soup(models: &Tensor) -> Result<Tensor> {
    models.mean(0)
}

/// Unified module encapsulating sharpness-aware and weight-averaging algorithms.
pub struct SharpnessAwareUnifiedModule {
    pub param_sam: Tensor,
    pub grad_at_w: Tensor,
    pub grad_at_pert: Tensor,
    pub golden_param_sam: Tensor,
    pub golden_epsilon_sam: Tensor,

    pub slow_p: Tensor,
    pub fast_p: Tensor,
    pub golden_slow_p: Tensor,
    pub golden_fast_p: Tensor,

    pub swa_avg: Tensor,
    pub ema_avg: Tensor,
    pub curr_param: Tensor,
    pub golden_swa: Tensor,
    pub golden_ema: Tensor,

    pub soup_models: Tensor,
    pub golden_soup: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl SharpnessAwareUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let param_sam = tensors
            .get("param_sam")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_sam".into()))?;
        let grad_at_w = tensors
            .get("grad_at_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_at_w".into()))?;
        let grad_at_pert = tensors
            .get("grad_at_pert")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_at_pert".into()))?;
        let golden_param_sam = tensors
            .get("golden_param_sam")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_sam".into()))?;
        let golden_epsilon_sam = tensors
            .get("golden_epsilon_sam")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_epsilon_sam".into()))?;

        let slow_p = tensors
            .get("slow_p")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing slow_p".into()))?;
        let fast_p = tensors
            .get("fast_p")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing fast_p".into()))?;
        let golden_slow_p = tensors
            .get("golden_slow_p")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_slow_p".into()))?;
        let golden_fast_p = tensors
            .get("golden_fast_p")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_fast_p".into()))?;

        let swa_avg = tensors
            .get("swa_avg")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing swa_avg".into()))?;
        let ema_avg = tensors
            .get("ema_avg")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing ema_avg".into()))?;
        let curr_param = tensors
            .get("curr_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing curr_param".into()))?;
        let golden_swa = tensors
            .get("golden_swa")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_swa".into()))?;
        let golden_ema = tensors
            .get("golden_ema")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_ema".into()))?;

        let soup_models = tensors
            .get("soup_models")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing soup_models".into()))?;
        let golden_soup = tensors
            .get("golden_soup")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_soup".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;

        Ok(Self {
            param_sam,
            grad_at_w,
            grad_at_pert,
            golden_param_sam,
            golden_epsilon_sam,
            slow_p,
            fast_p,
            golden_slow_p,
            golden_fast_p,
            swa_avg,
            ema_avg,
            curr_param,
            golden_swa,
            golden_ema,
            soup_models,
            golden_soup,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // 1. SAM Invariant
        let (p_sam, eps_sam) = step_sam(
            &self.param_sam,
            &self.grad_at_w,
            &self.grad_at_pert,
            0.05,
            0.01,
            0.001,
            1e-12,
        )?;
        let err_p_sam = max_abs_diff(&p_sam, &self.golden_param_sam)?;
        let err_eps_sam = max_abs_diff(&eps_sam, &self.golden_epsilon_sam)?;
        let err_sam = err_p_sam.max(err_eps_sam);
        if err_sam >= atol {
            candle_core::bail!("SAM invariant failed: err={:e} >= {:e}", err_sam, atol);
        }
        errors.insert("sam_err".to_string(), err_sam);

        // 2. Lookahead Invariant
        let (slow_out, fast_out) = step_lookahead(&self.slow_p, &self.fast_p, 0.5)?;
        let err_slow = max_abs_diff(&slow_out, &self.golden_slow_p)?;
        let err_fast = max_abs_diff(&fast_out, &self.golden_fast_p)?;
        let err_lookahead = err_slow.max(err_fast);
        if err_lookahead >= atol {
            candle_core::bail!(
                "Lookahead invariant failed: err={:e} >= {:e}",
                err_lookahead,
                atol
            );
        }
        errors.insert("lookahead_err".to_string(), err_lookahead);

        // 3. SWA & EMA Invariant
        let (new_swa, new_ema) =
            step_swa_ema(&self.swa_avg, &self.ema_avg, &self.curr_param, 4, 0.99)?;
        let err_swa = max_abs_diff(&new_swa, &self.golden_swa)?;
        let err_ema = max_abs_diff(&new_ema, &self.golden_ema)?;
        let err_avg = err_swa.max(err_ema);
        if err_avg >= atol {
            candle_core::bail!("SWA/EMA invariant failed: err={:e} >= {:e}", err_avg, atol);
        }
        errors.insert("swa_ema_err".to_string(), err_avg);

        // 4. Model Soup Invariant
        let soup_out = uniform_model_soup(&self.soup_models)?;
        let err_soup = max_abs_diff(&soup_out, &self.golden_soup)?;
        if err_soup >= atol {
            candle_core::bail!(
                "Model Soup invariant failed: err={:e} >= {:e}",
                err_soup,
                atol
            );
        }
        errors.insert("soup_err".to_string(), err_soup);

        // 5. Linear Anchor Invariant
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
