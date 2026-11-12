//! Candle Implementation for Category 04 Topic 09 (Subfolder 02 Topic 02):
//! Multi-Objective Pareto, MGDA, Tchebycheff, and Gradient Surgery.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Multiple Gradient Descent Algorithm (MGDA - Désidéri 2012; Sener & Koltun 2018).
//! 2. Projecting Conflicting Gradients (PCGrad - Yu et al. 2020).
//! 3. Augmented Tchebycheff Scalarization for Non-Convex Pareto Fronts (Miettinen 1999).
//! 4. Linear Multi-Task Scalarization.

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

/// Compute 2-task MGDA min-norm convex combination gradient and parameter step.
pub fn step_mgda_2task(
    param: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    lr: f64,
    eps: f64,
) -> Result<(Tensor, Tensor, f64)> {
    let p_vec = param.to_vec1::<f32>()?;
    let g1_vec = g1.to_vec1::<f32>()?;
    let g2_vec = g2.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut u_norm_sq = 0.0f64;
    let mut num = 0.0f64;
    for i in 0..d {
        let u_i = (g1_vec[i] as f64) - (g2_vec[i] as f64);
        u_norm_sq += u_i * u_i;
        num += ((g2_vec[i] as f64) - (g1_vec[i] as f64)) * (g2_vec[i] as f64);
    }

    let alpha = (num / (u_norm_sq + eps)).clamp(0.0, 1.0);

    let mut g_mgda = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];
    for i in 0..d {
        let gm = (alpha * (g1_vec[i] as f64) + (1.0 - alpha) * (g2_vec[i] as f64)) as f32;
        g_mgda[i] = gm;
        p_next[i] = p_vec[i] - (lr as f32) * gm;
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(p_next, d, dev)?,
        Tensor::from_vec(g_mgda, d, dev)?,
        alpha,
    ))
}

/// Perform PCGrad conflicting gradient projection and parameter step.
pub fn step_pcgrad_2task(
    param: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    lr: f64,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let p_vec = param.to_vec1::<f32>()?;
    let g1_vec = g1.to_vec1::<f32>()?;
    let g2_vec = g2.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut dot_12 = 0.0f64;
    let mut norm_sq_1 = 0.0f64;
    let mut norm_sq_2 = 0.0f64;
    for i in 0..d {
        let a = g1_vec[i] as f64;
        let b = g2_vec[i] as f64;
        dot_12 += a * b;
        norm_sq_1 += a * a;
        norm_sq_2 += b * b;
    }

    let mut g_pc = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    if dot_12 < 0.0 {
        let scale_1 = dot_12 / (norm_sq_2 + eps);
        let scale_2 = dot_12 / (norm_sq_1 + eps);
        for i in 0..d {
            let g1_proj = (g1_vec[i] as f64) - scale_1 * (g2_vec[i] as f64);
            let g2_proj = (g2_vec[i] as f64) - scale_2 * (g1_vec[i] as f64);
            let gp = (0.5 * (g1_proj + g2_proj)) as f32;
            g_pc[i] = gp;
            p_next[i] = p_vec[i] - (lr as f32) * gp;
        }
    } else {
        for i in 0..d {
            let gp = (0.5 * ((g1_vec[i] as f64) + (g2_vec[i] as f64))) as f32;
            g_pc[i] = gp;
            p_next[i] = p_vec[i] - (lr as f32) * gp;
        }
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(p_next, d, dev)?,
        Tensor::from_vec(g_pc, d, dev)?,
    ))
}

/// Perform Augmented Tchebycheff scalarization gradient step.
pub fn step_tchebycheff(
    param: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    losses: &Tensor,
    z_star: &Tensor,
    weights: &Tensor,
    rho: f64,
    lr: f64,
) -> Result<(Tensor, Tensor)> {
    let p_vec = param.to_vec1::<f32>()?;
    let g1_vec = g1.to_vec1::<f32>()?;
    let g2_vec = g2.to_vec1::<f32>()?;
    let l_vec = losses.to_vec1::<f32>()?;
    let z_vec = z_star.to_vec1::<f32>()?;
    let w_vec = weights.to_vec1::<f32>()?;
    let d = p_vec.len();

    let diff0 = (w_vec[0] as f64) * ((l_vec[0] as f64) - (z_vec[0] as f64));
    let diff1 = (w_vec[1] as f64) * ((l_vec[1] as f64) - (z_vec[1] as f64));

    let idx_max = if diff0 >= diff1 { 0 } else { 1 };

    let mut ds_dl = [rho * (w_vec[0] as f64), rho * (w_vec[1] as f64)];
    ds_dl[idx_max] += w_vec[idx_max] as f64;

    let mut g_tch = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    for i in 0..d {
        let gt = (ds_dl[0] * (g1_vec[i] as f64) + ds_dl[1] * (g2_vec[i] as f64)) as f32;
        g_tch[i] = gt;
        p_next[i] = p_vec[i] - (lr as f32) * gt;
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(p_next, d, dev)?,
        Tensor::from_vec(g_tch, d, dev)?,
    ))
}

/// Perform linear multi-task scalarization gradient step.
pub fn step_linear_scalarization(
    param: &Tensor,
    g1: &Tensor,
    g2: &Tensor,
    weights: &Tensor,
    lr: f64,
) -> Result<(Tensor, Tensor)> {
    let p_vec = param.to_vec1::<f32>()?;
    let g1_vec = g1.to_vec1::<f32>()?;
    let g2_vec = g2.to_vec1::<f32>()?;
    let w_vec = weights.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut g_lin = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    for i in 0..d {
        let gl = ((w_vec[0] as f64) * (g1_vec[i] as f64) + (w_vec[1] as f64) * (g2_vec[i] as f64))
            as f32;
        g_lin[i] = gl;
        p_next[i] = p_vec[i] - (lr as f32) * gl;
    }

    let dev = param.device();
    Ok((
        Tensor::from_vec(p_next, d, dev)?,
        Tensor::from_vec(g_lin, d, dev)?,
    ))
}

/// Unified module encapsulating multi-objective optimizers for parity verification.
pub struct MultiObjectiveUnifiedModule {
    pub param: Tensor,
    pub g1: Tensor,
    pub g2: Tensor,
    pub losses: Tensor,
    pub z_star: Tensor,
    pub weights_tch: Tensor,
    pub weights_lin: Tensor,
    pub anchor: LinearAnchor,
    pub golden_alpha_mgda: f64,
    pub golden_g_mgda: Tensor,
    pub golden_param_mgda: Tensor,
    pub golden_g_pc: Tensor,
    pub golden_param_pc: Tensor,
    pub golden_g_tch: Tensor,
    pub golden_param_tch: Tensor,
    pub golden_g_lin: Tensor,
    pub golden_param_lin: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl MultiObjectiveUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let param = tensors
            .get("param")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param".into()))?;
        let g1 = tensors
            .get("g1")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing g1".into()))?;
        let g2 = tensors
            .get("g2")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing g2".into()))?;
        let losses = tensors
            .get("L")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing L".into()))?;
        let z_star = tensors
            .get("z_star")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing z_star".into()))?;
        let weights_tch = tensors
            .get("weights_tch")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing weights_tch".into()))?;
        let weights_lin = tensors
            .get("weights_lin")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing weights_lin".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_alpha_mgda = tensors
            .get("golden_alpha_mgda")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_alpha_mgda".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let golden_g_mgda = tensors
            .get("golden_g_mgda")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_g_mgda".into()))?;
        let golden_param_mgda = tensors
            .get("golden_param_mgda")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_mgda".into()))?;
        let golden_g_pc = tensors
            .get("golden_g_pc")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_g_pc".into()))?;
        let golden_param_pc = tensors
            .get("golden_param_pc")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_pc".into()))?;
        let golden_g_tch = tensors
            .get("golden_g_tch")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_g_tch".into()))?;
        let golden_param_tch = tensors
            .get("golden_param_tch")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_tch".into()))?;
        let golden_g_lin = tensors
            .get("golden_g_lin")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_g_lin".into()))?;
        let golden_param_lin = tensors
            .get("golden_param_lin")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_lin".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            param,
            g1,
            g2,
            losses,
            z_star,
            weights_tch,
            weights_lin,
            anchor,
            golden_alpha_mgda,
            golden_g_mgda,
            golden_param_mgda,
            golden_g_pc,
            golden_param_pc,
            golden_g_tch,
            golden_param_tch,
            golden_g_lin,
            golden_param_lin,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: MGDA
        let (p_mgda, g_mgda, alpha) =
            step_mgda_2task(&self.param, &self.g1, &self.g2, 0.01, 1e-12)?;
        let err_alpha = (alpha - self.golden_alpha_mgda).abs();
        let err_g_mgda = max_abs_diff(&g_mgda, &self.golden_g_mgda)?;
        let err_p_mgda = max_abs_diff(&p_mgda, &self.golden_param_mgda)?;
        let err_mgda = err_alpha.max(err_g_mgda).max(err_p_mgda);
        if err_mgda >= atol {
            candle_core::bail!("MGDA invariant failed: err={:e} >= {:e}", err_mgda, atol);
        }
        errors.insert("mgda_err".to_string(), err_mgda);

        // Invariant 2: PCGrad
        let (p_pc, g_pc) = step_pcgrad_2task(&self.param, &self.g1, &self.g2, 0.01, 1e-12)?;
        let err_g_pc = max_abs_diff(&g_pc, &self.golden_g_pc)?;
        let err_p_pc = max_abs_diff(&p_pc, &self.golden_param_pc)?;
        let err_pc = err_g_pc.max(err_p_pc);
        if err_pc >= atol {
            candle_core::bail!("PCGrad invariant failed: err={:e} >= {:e}", err_pc, atol);
        }
        errors.insert("pcgrad_err".to_string(), err_pc);

        // Invariant 3: Tchebycheff
        let (p_tch, g_tch) = step_tchebycheff(
            &self.param,
            &self.g1,
            &self.g2,
            &self.losses,
            &self.z_star,
            &self.weights_tch,
            0.05,
            0.01,
        )?;
        let err_g_tch = max_abs_diff(&g_tch, &self.golden_g_tch)?;
        let err_p_tch = max_abs_diff(&p_tch, &self.golden_param_tch)?;
        let err_tch = err_g_tch.max(err_p_tch);
        if err_tch >= atol {
            candle_core::bail!(
                "Tchebycheff invariant failed: err={:e} >= {:e}",
                err_tch,
                atol
            );
        }
        errors.insert("tchebycheff_err".to_string(), err_tch);

        // Invariant 4: Linear Scalarization
        let (p_lin, g_lin) =
            step_linear_scalarization(&self.param, &self.g1, &self.g2, &self.weights_lin, 0.01)?;
        let err_g_lin = max_abs_diff(&g_lin, &self.golden_g_lin)?;
        let err_p_lin = max_abs_diff(&p_lin, &self.golden_param_lin)?;
        let err_lin = err_g_lin.max(err_p_lin);
        if err_lin >= atol {
            candle_core::bail!(
                "Linear scalarization invariant failed: err={:e} >= {:e}",
                err_lin,
                atol
            );
        }
        errors.insert("linear_scalarization_err".to_string(), err_lin);

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
