//! Candle Implementation for Category 04 Topic 06:
//! Schedule-Free and Parameter-Free Optimisers.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Schedule-Free AdamW (Defazio et al. 2024): Continuous iterate averaging eliminating LR schedules.
//! 2. Polyak Step-Size: Exact optimal step scaling using loss gap (f - f*).
//! 3. Barzilai-Borwein Spectral Step-Size (BB1): Non-monotonic two-point secant step estimation.
//! 4. D-Adaptation: Online distance-to-solution tracking.

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

/// Perform one Schedule-Free AdamW update step.
pub fn step_schedule_free_adamw(
    x: &Tensor,
    z: &Tensor,
    y: &Tensor,
    grad: &Tensor,
    m: &Tensor,
    v: &Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
    step: usize,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let x_vec = x.to_vec1::<f32>()?;
    let z_vec = z.to_vec1::<f32>()?;
    let y_vec = y.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let m_vec = m.to_vec1::<f32>()?;
    let v_vec = v.to_vec1::<f32>()?;
    let d = x_vec.len();

    let bias_v = 1.0 - beta2.powi(step as i32);
    let c_prime = 1.0 / ((step + 1) as f64);
    let c_next = 1.0 / ((step + 2) as f64);

    let mut m_next = vec![0.0f32; d];
    let mut v_next = vec![0.0f32; d];
    let mut z_next = vec![0.0f32; d];
    let mut x_next = vec![0.0f32; d];
    let mut y_next = vec![0.0f32; d];

    for j in 0..d {
        let g = g_vec[j] as f64;
        let m_val = beta1 * (m_vec[j] as f64) + (1.0 - beta1) * g;
        let v_val = beta2 * (v_vec[j] as f64) + (1.0 - beta2) * (g * g);
        let v_hat = v_val / bias_v;

        let u = (m_val / (v_hat.sqrt() + eps)) + weight_decay * (y_vec[j] as f64);
        let z_val = (z_vec[j] as f64) - lr * u;
        let x_val = (1.0 - c_prime) * (x_vec[j] as f64) + c_prime * z_val;
        let y_val = (1.0 - c_next) * z_val + c_next * x_val;

        m_next[j] = m_val as f32;
        v_next[j] = v_val as f32;
        z_next[j] = z_val as f32;
        x_next[j] = x_val as f32;
        y_next[j] = y_val as f32;
    }

    let dev = x.device();
    Ok((
        Tensor::from_vec(x_next, d, dev)?,
        Tensor::from_vec(z_next, d, dev)?,
        Tensor::from_vec(y_next, d, dev)?,
        Tensor::from_vec(m_next, d, dev)?,
        Tensor::from_vec(v_next, d, dev)?,
    ))
}

/// Perform one step using Polyak's optimal step-size: gamma = (f - f*) / ||g||^2.
pub fn step_polyak(
    param: &Tensor,
    grad: &Tensor,
    f_val: f64,
    f_opt: f64,
    eps: f64,
) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut norm_sq = 0.0f64;
    for &g in &g_vec {
        let g_f64 = g as f64;
        norm_sq += g_f64 * g_f64;
    }

    let gamma = (f_val - f_opt) / (norm_sq + eps);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = ((p_vec[j] as f64) - gamma * (g_vec[j] as f64)) as f32;
    }

    Tensor::from_vec(out, d, param.device())
}

/// Perform one step using Barzilai-Borwein BB1 spectral step-size: gamma = (s^T s) / (s^T y).
pub fn step_barzilai_borwein(
    param: &Tensor,
    grad: &Tensor,
    s: &Tensor,
    y: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let s_vec = s.to_vec1::<f32>()?;
    let y_vec = y.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut s_dot_s = 0.0f64;
    let mut s_dot_y = 0.0f64;
    for j in 0..d {
        let s_val = s_vec[j] as f64;
        let y_val = y_vec[j] as f64;
        s_dot_s += s_val * s_val;
        s_dot_y += s_val * y_val;
    }

    let gamma = s_dot_s / (s_dot_y + eps);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = ((p_vec[j] as f64) - gamma * (g_vec[j] as f64)) as f32;
    }

    Tensor::from_vec(out, d, param.device())
}

/// Perform one D-Adaptation step with distance-to-solution tracking.
pub fn step_d_adaptation(
    param: &Tensor,
    grad: &Tensor,
    d_est: f64,
    lr0: f64,
    eps: f64,
) -> Result<(Tensor, f64)> {
    let p_vec = param.to_vec1::<f32>()?;
    let g_vec = grad.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut l1_norm = 0.0f64;
    let mut l2_norm_sq = 0.0f64;
    for &g in &g_vec {
        let abs_g = (g as f64).abs();
        l1_norm += abs_g;
        l2_norm_sq += abs_g * abs_g;
    }
    let l2_norm = l2_norm_sq.sqrt();

    let d_next = (d_est + 0.05 * (l1_norm / ((d as f64).sqrt()) - 1.0)).max(0.1);
    let eff_lr = lr0 * d_next;

    let mut out = vec![0.0f32; d];
    for j in 0..d {
        let step = eff_lr * (g_vec[j] as f64) / (l2_norm + eps);
        out[j] = ((p_vec[j] as f64) - step) as f32;
    }

    Ok((Tensor::from_vec(out, d, param.device())?, d_next))
}

/// Unified module encapsulating schedule-free and parameter-free optimisers in Candle.
pub struct ScheduleFreeUnifiedModule {
    pub sf_x: Tensor,
    pub sf_z: Tensor,
    pub sf_y: Tensor,
    pub sf_grad: Tensor,
    pub sf_m: Tensor,
    pub sf_v: Tensor,
    pub polyak_param: Tensor,
    pub polyak_grad: Tensor,
    pub f_val: f64,
    pub f_opt: f64,
    pub bb_param: Tensor,
    pub bb_grad: Tensor,
    pub bb_s: Tensor,
    pub bb_y: Tensor,
    pub d_param: Tensor,
    pub d_grad: Tensor,
    pub d_est: f64,
    pub anchor: LinearAnchor,
    pub golden_sf_x: Tensor,
    pub golden_sf_z: Tensor,
    pub golden_sf_y: Tensor,
    pub golden_polyak_param: Tensor,
    pub golden_bb_param: Tensor,
    pub golden_d_adapt_param: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl ScheduleFreeUnifiedModule {
    pub fn load_from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let data = std::fs::read(path).map_err(candle_core::Error::wrap)?;
        let tensors = candle_core::safetensors::load_buffer(&data, device)?;

        let sf_x = tensors
            .get("sf_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_x".into()))?;
        let sf_z = tensors
            .get("sf_z")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_z".into()))?;
        let sf_y = tensors
            .get("sf_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_y".into()))?;
        let sf_grad = tensors
            .get("sf_grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_grad".into()))?;
        let sf_m = tensors
            .get("sf_m")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_m".into()))?;
        let sf_v = tensors
            .get("sf_v")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sf_v".into()))?;

        let polyak_param = tensors
            .get("polyak_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing polyak_param".into()))?;
        let polyak_grad = tensors
            .get("polyak_grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing polyak_grad".into()))?;
        let f_val = tensors
            .get("f_val")
            .ok_or_else(|| candle_core::Error::Msg("Missing f_val".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let f_opt = tensors
            .get("f_opt")
            .ok_or_else(|| candle_core::Error::Msg("Missing f_opt".into()))?
            .to_vec1::<f32>()?[0] as f64;

        let bb_param = tensors
            .get("bb_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing bb_param".into()))?;
        let bb_grad = tensors
            .get("bb_grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing bb_grad".into()))?;
        let bb_s = tensors
            .get("bb_s")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing bb_s".into()))?;
        let bb_y = tensors
            .get("bb_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing bb_y".into()))?;

        let d_param = tensors
            .get("d_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing d_param".into()))?;
        let d_grad = tensors
            .get("d_grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing d_grad".into()))?;
        let d_est = tensors
            .get("d_est")
            .ok_or_else(|| candle_core::Error::Msg("Missing d_est".into()))?
            .to_vec1::<f32>()?[0] as f64;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_sf_x = tensors
            .get("golden_sf_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sf_x".into()))?;
        let golden_sf_z = tensors
            .get("golden_sf_z")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sf_z".into()))?;
        let golden_sf_y = tensors
            .get("golden_sf_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sf_y".into()))?;
        let golden_polyak_param = tensors
            .get("golden_polyak_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_polyak_param".into()))?;
        let golden_bb_param = tensors
            .get("golden_bb_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_bb_param".into()))?;
        let golden_d_adapt_param = tensors
            .get("golden_d_adapt_param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_d_adapt_param".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            sf_x,
            sf_z,
            sf_y,
            sf_grad,
            sf_m,
            sf_v,
            polyak_param,
            polyak_grad,
            f_val,
            f_opt,
            bb_param,
            bb_grad,
            bb_s,
            bb_y,
            d_param,
            d_grad,
            d_est,
            anchor,
            golden_sf_x,
            golden_sf_z,
            golden_sf_y,
            golden_polyak_param,
            golden_bb_param,
            golden_d_adapt_param,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: Schedule-Free AdamW
        let (x_next, z_next, y_next, _, _) = step_schedule_free_adamw(
            &self.sf_x,
            &self.sf_z,
            &self.sf_y,
            &self.sf_grad,
            &self.sf_m,
            &self.sf_v,
            0.01,
            0.9,
            0.999,
            0.01,
            1,
            1e-8,
        )?;
        let err_x = max_abs_diff(&x_next, &self.golden_sf_x)?;
        let err_z = max_abs_diff(&z_next, &self.golden_sf_z)?;
        let err_y = max_abs_diff(&y_next, &self.golden_sf_y)?;
        let err_sf = err_x.max(err_z).max(err_y);
        if err_sf >= atol {
            candle_core::bail!(
                "Schedule-Free invariant failed: err={:e} >= {:e}",
                err_sf,
                atol
            );
        }
        errors.insert("schedule_free_err".to_string(), err_sf);

        // Invariant 2: Polyak
        let p_polyak = step_polyak(
            &self.polyak_param,
            &self.polyak_grad,
            self.f_val,
            self.f_opt,
            1e-8,
        )?;
        let err_polyak = max_abs_diff(&p_polyak, &self.golden_polyak_param)?;
        if err_polyak >= atol {
            candle_core::bail!(
                "Polyak invariant failed: err={:e} >= {:e}",
                err_polyak,
                atol
            );
        }
        errors.insert("polyak_err".to_string(), err_polyak);

        // Invariant 3: Barzilai-Borwein
        let p_bb =
            step_barzilai_borwein(&self.bb_param, &self.bb_grad, &self.bb_s, &self.bb_y, 1e-8)?;
        let err_bb = max_abs_diff(&p_bb, &self.golden_bb_param)?;
        if err_bb >= atol {
            candle_core::bail!(
                "Barzilai-Borwein invariant failed: err={:e} >= {:e}",
                err_bb,
                atol
            );
        }
        errors.insert("bb_err".to_string(), err_bb);

        // Invariant 4: D-Adaptation
        let (p_dadapt, _) = step_d_adaptation(&self.d_param, &self.d_grad, self.d_est, 0.01, 1e-8)?;
        let err_dadapt = max_abs_diff(&p_dadapt, &self.golden_d_adapt_param)?;
        if err_dadapt >= atol {
            candle_core::bail!(
                "D-Adaptation invariant failed: err={:e} >= {:e}",
                err_dadapt,
                atol
            );
        }
        errors.insert("d_adapt_err".to_string(), err_dadapt);

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
