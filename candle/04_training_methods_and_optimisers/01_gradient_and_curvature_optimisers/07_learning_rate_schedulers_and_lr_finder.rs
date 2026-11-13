//! Candle Implementation for Category 04 Topic 07:
//! Learning Rate Schedulers and LR Finder.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Cosine Annealing with Linear Warmup (Loshchilov & Hutter 2016).
//! 2. Warmup-Stable-Decay (WSD - Hu et al. 2024 MiniCPM): Horizon-independent pretraining schedule.
//! 3. OneCycleLR (Smith 2017): Dynamic cyclical schedule with asymmetric momentum.
//! 4. AutoClip (Seetharaman et al. 2020): Adaptive gradient clipping by historical percentiles.

use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;
use std::f64::consts::PI;
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

/// Compute learning rate for Cosine Annealing with Linear Warmup.
pub fn schedule_cosine_warmup(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    eta_min: f64,
    eta_max: f64,
) -> f64 {
    if step <= warmup_steps {
        return eta_min + (step as f64 / warmup_steps.max(1) as f64) * (eta_max - eta_min);
    }
    let ratio = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
    eta_min + 0.5 * (eta_max - eta_min) * (1.0 + (PI * ratio).cos())
}

/// Compute learning rate for Warmup-Stable-Decay (WSD).
pub fn schedule_wsd(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    decay_steps: usize,
    eta_min: f64,
    eta_max: f64,
) -> f64 {
    if step <= warmup_steps {
        return eta_min + (step as f64 / warmup_steps.max(1) as f64) * (eta_max - eta_min);
    }
    let decay_start = total_steps - decay_steps;
    if step <= decay_start {
        return eta_max;
    }
    let ratio = (step - decay_start) as f64 / decay_steps.max(1) as f64;
    eta_min + 0.5 * (eta_max - eta_min) * (1.0 + (PI * ratio).cos())
}

/// Compute learning rate for OneCycleLR.
pub fn schedule_onecycle(
    step: usize,
    total_steps: usize,
    peak_steps: usize,
    eta_max: f64,
    div_factor: f64,
    final_div_factor: f64,
) -> f64 {
    let eta_init = eta_max / div_factor;
    let eta_min = eta_init / final_div_factor;
    if step <= peak_steps {
        let ratio = step as f64 / peak_steps.max(1) as f64;
        return eta_init + 0.5 * (eta_max - eta_init) * (1.0 - (PI * ratio).cos());
    }
    let ratio = (step - peak_steps) as f64 / (total_steps - peak_steps).max(1) as f64;
    eta_min + 0.5 * (eta_max - eta_min) * (1.0 + (PI * ratio).cos())
}

/// Compute percentile with linear interpolation matching numpy default.
pub fn compute_percentile(data: &[f32], q: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0] as f64;
    }
    let idx = (q / 100.0) * ((n - 1) as f64);
    let low = idx.floor() as usize;
    let high = idx.ceil() as usize;
    let frac = idx - (low as f64);
    if low >= n - 1 {
        sorted[n - 1] as f64
    } else {
        (sorted[low] as f64) + frac * ((sorted[high] as f64) - (sorted[low] as f64))
    }
}

/// Perform AutoClip adaptive gradient percentile clipping and parameter step.
pub fn step_autoclip(
    param: &Tensor,
    grad: &Tensor,
    norm_history: &Tensor,
    percentile: f64,
    lr: f64,
    eps: f64,
) -> Result<(Tensor, Tensor, f64)> {
    let param_vec = param.to_vec1::<f32>()?;
    let grad_vec = grad.to_vec1::<f32>()?;
    let hist_vec = norm_history.to_vec1::<f32>()?;
    let d = param_vec.len();

    let tau = compute_percentile(&hist_vec, percentile);
    let grad_norm = (grad_vec
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>())
    .sqrt();

    let clip_ratio = if tau < (grad_norm + eps) {
        tau / (grad_norm + eps)
    } else {
        1.0
    };

    let mut clipped_grad = vec![0.0f32; d];
    let mut param_next = vec![0.0f32; d];

    for i in 0..d {
        let cg = (grad_vec[i] as f64 * clip_ratio) as f32;
        clipped_grad[i] = cg;
        param_next[i] = param_vec[i] - (lr as f32) * cg;
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(param_next, d, dev)?,
        Tensor::from_vec(clipped_grad, d, dev)?,
        tau,
    ))
}

/// Unified module encapsulating learning rate schedulers and AutoClip for parity verification.
pub struct LRSchedulersUnifiedModule {
    pub param_cosine: Tensor,
    pub grad_cosine: Tensor,
    pub param_wsd: Tensor,
    pub grad_wsd: Tensor,
    pub param_onecycle: Tensor,
    pub grad_onecycle: Tensor,
    pub param_autoclip: Tensor,
    pub grad_autoclip: Tensor,
    pub norm_history: Tensor,
    pub anchor: LinearAnchor,
    pub golden_lr_cosine: f64,
    pub golden_param_cosine: Tensor,
    pub golden_lr_wsd: f64,
    pub golden_param_wsd: Tensor,
    pub golden_lr_onecycle: f64,
    pub golden_param_onecycle: Tensor,
    pub golden_clipped_grad: Tensor,
    pub golden_param_autoclip: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl LRSchedulersUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let param_cosine = tensors
            .get("param_cosine")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_cosine".into()))?;
        let grad_cosine = tensors
            .get("grad_cosine")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_cosine".into()))?;
        let param_wsd = tensors
            .get("param_wsd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_wsd".into()))?;
        let grad_wsd = tensors
            .get("grad_wsd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_wsd".into()))?;
        let param_onecycle = tensors
            .get("param_onecycle")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_onecycle".into()))?;
        let grad_onecycle = tensors
            .get("grad_onecycle")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_onecycle".into()))?;
        let param_autoclip = tensors
            .get("param_autoclip")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_autoclip".into()))?;
        let grad_autoclip = tensors
            .get("grad_autoclip")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad_autoclip".into()))?;
        let norm_history = tensors
            .get("norm_history")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing norm_history".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_lr_cosine = tensors
            .get("golden_lr_cosine")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_lr_cosine".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let golden_param_cosine = tensors
            .get("golden_param_cosine")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_cosine".into()))?;
        let golden_lr_wsd = tensors
            .get("golden_lr_wsd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_lr_wsd".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let golden_param_wsd = tensors
            .get("golden_param_wsd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_wsd".into()))?;
        let golden_lr_onecycle = tensors
            .get("golden_lr_onecycle")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_lr_onecycle".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let golden_param_onecycle = tensors
            .get("golden_param_onecycle")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_onecycle".into()))?;
        let golden_clipped_grad = tensors
            .get("golden_clipped_grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_clipped_grad".into()))?;
        let golden_param_autoclip = tensors
            .get("golden_param_autoclip")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_autoclip".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            param_cosine,
            grad_cosine,
            param_wsd,
            grad_wsd,
            param_onecycle,
            grad_onecycle,
            param_autoclip,
            grad_autoclip,
            norm_history,
            anchor,
            golden_lr_cosine,
            golden_param_cosine,
            golden_lr_wsd,
            golden_param_wsd,
            golden_lr_onecycle,
            golden_param_onecycle,
            golden_clipped_grad,
            golden_param_autoclip,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: Cosine Annealing with Warmup
        let lr_cos = schedule_cosine_warmup(30, 100, 10, 1e-5, 0.01);
        let err_lr_cos = (lr_cos - self.golden_lr_cosine).abs();
        let p_cos_vec: Vec<f32> = self
            .param_cosine
            .to_vec1::<f32>()?
            .iter()
            .zip(self.grad_cosine.to_vec1::<f32>()?.iter())
            .map(|(&p, &g)| p - (lr_cos as f32) * g)
            .collect();
        let p_cos = Tensor::from_vec(
            p_cos_vec,
            self.param_cosine.dim(0)?,
            self.param_cosine.device(),
        )?;
        let err_p_cos = max_abs_diff(&p_cos, &self.golden_param_cosine)?;
        let err_cos = err_lr_cos.max(err_p_cos);
        if err_cos >= atol {
            candle_core::bail!("Cosine invariant failed: err={:e} >= {:e}", err_cos, atol);
        }
        errors.insert("cosine_err".to_string(), err_cos);

        // Invariant 2: WSD
        let lr_wsd = schedule_wsd(85, 100, 10, 20, 1e-4, 0.01);
        let err_lr_wsd = (lr_wsd - self.golden_lr_wsd).abs();
        let p_wsd_vec: Vec<f32> = self
            .param_wsd
            .to_vec1::<f32>()?
            .iter()
            .zip(self.grad_wsd.to_vec1::<f32>()?.iter())
            .map(|(&p, &g)| p - (lr_wsd as f32) * g)
            .collect();
        let p_wsd = Tensor::from_vec(p_wsd_vec, self.param_wsd.dim(0)?, self.param_wsd.device())?;
        let err_p_wsd = max_abs_diff(&p_wsd, &self.golden_param_wsd)?;
        let err_wsd = err_lr_wsd.max(err_p_wsd);
        if err_wsd >= atol {
            candle_core::bail!("WSD invariant failed: err={:e} >= {:e}", err_wsd, atol);
        }
        errors.insert("wsd_err".to_string(), err_wsd);

        // Invariant 3: OneCycleLR
        let lr_oc = schedule_onecycle(40, 100, 30, 0.01, 25.0, 1000.0);
        let err_lr_oc = (lr_oc - self.golden_lr_onecycle).abs();
        let p_oc_vec: Vec<f32> = self
            .param_onecycle
            .to_vec1::<f32>()?
            .iter()
            .zip(self.grad_onecycle.to_vec1::<f32>()?.iter())
            .map(|(&p, &g)| p - (lr_oc as f32) * g)
            .collect();
        let p_oc = Tensor::from_vec(
            p_oc_vec,
            self.param_onecycle.dim(0)?,
            self.param_onecycle.device(),
        )?;
        let err_p_oc = max_abs_diff(&p_oc, &self.golden_param_onecycle)?;
        let err_oc = err_lr_oc.max(err_p_oc);
        if err_oc >= atol {
            candle_core::bail!("OneCycle invariant failed: err={:e} >= {:e}", err_oc, atol);
        }
        errors.insert("onecycle_err".to_string(), err_oc);

        // Invariant 4: AutoClip
        let (p_ac, clipped_g, _) = step_autoclip(
            &self.param_autoclip,
            &self.grad_autoclip,
            &self.norm_history,
            90.0,
            0.01,
            1e-8,
        )?;
        let err_cg = max_abs_diff(clipped_grad_ref(&clipped_g), &self.golden_clipped_grad)?;
        let err_p_ac = max_abs_diff(&p_ac, &self.golden_param_autoclip)?;
        let err_ac = err_cg.max(err_p_ac);
        if err_ac >= atol {
            candle_core::bail!("AutoClip invariant failed: err={:e} >= {:e}", err_ac, atol);
        }
        errors.insert("autoclip_err".to_string(), err_ac);

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

fn clipped_grad_ref(t: &Tensor) -> &Tensor {
    t
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = a.sub(b)?.abs()?;
    let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(max as f64)
}
