//! Candle Implementation for Category 04 Topic 10 (Subfolder 02 Topic 03):
//! Coordinate, Alternating Projections, and Majorization-Minimization.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Dykstra's Alternating Projections onto intersection of convex sets.
//! 2. Block Coordinate Descent (BCD) with cyclic coordinate updates.
//! 3. Iteratively Reweighted Least Squares (IRLS) for robust L1/Lp norms.
//! 4. Concave-Convex Procedure (CCCP) and Majorization-Minimization (MM).

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

/// Project onto halfspace <normal, x> <= offset.
pub fn proj_halfspace(x: &[f32], normal: &[f32], offset: f64) -> Vec<f32> {
    let d = x.len();
    let mut dot = 0.0f64;
    let mut norm_sq = 0.0f64;
    for i in 0..d {
        dot += (x[i] as f64) * (normal[i] as f64);
        norm_sq += (normal[i] as f64) * (normal[i] as f64);
    }
    if dot <= offset {
        return x.to_vec();
    }
    let factor = (dot - offset) / norm_sq;
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        out[i] = ((x[i] as f64) - factor * (normal[i] as f64)) as f32;
    }
    out
}

/// Project onto box [lower, upper]^d.
pub fn proj_box(x: &[f32], lower: f64, upper: f64) -> Vec<f32> {
    x.iter()
        .map(|&v| (v as f64).clamp(lower, upper) as f32)
        .collect()
}

/// Perform one full step of Dykstra's Alternating Projections algorithm.
pub fn step_dykstra(
    y_prev: &Tensor,
    p_prev: &Tensor,
    q_prev: &Tensor,
    normal: &Tensor,
    offset: f64,
    lower: f64,
    upper: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let y_vec = y_prev.to_vec1::<f32>()?;
    let p_vec = p_prev.to_vec1::<f32>()?;
    let q_vec = q_prev.to_vec1::<f32>()?;
    let n_vec = normal.to_vec1::<f32>()?;
    let d = y_vec.len();

    // Step 1: y_prev + p_prev
    let mut yp = vec![0.0f32; d];
    for i in 0..d {
        yp[i] = y_vec[i] + p_vec[i];
    }
    let x_curr = proj_halfspace(&yp, &n_vec, offset);

    // p_curr = y_prev + p_prev - x_curr
    let mut p_curr = vec![0.0f32; d];
    for i in 0..d {
        p_curr[i] = yp[i] - x_curr[i];
    }

    // Step 2: x_curr + q_prev
    let mut xq = vec![0.0f32; d];
    for i in 0..d {
        xq[i] = x_curr[i] + q_vec[i];
    }
    let y_curr = proj_box(&xq, lower, upper);

    // q_curr = x_curr + q_prev - y_curr
    let mut q_curr = vec![0.0f32; d];
    for i in 0..d {
        q_curr[i] = xq[i] - y_curr[i];
    }

    let dev = y_prev.device();
    Ok((
        Tensor::from_vec(x_curr, d, dev)?,
        Tensor::from_vec(y_curr, d, dev)?,
        Tensor::from_vec(p_curr, d, dev)?,
        Tensor::from_vec(q_curr, d, dev)?,
    ))
}

/// Block Coordinate Descent update step: (b - coupling * param) / a_diag.
pub fn step_bcd(param: &Tensor, b: &Tensor, a_diag: &Tensor, coupling: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let a_vec = a_diag.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        out[i] = (((b_vec[i] as f64) - coupling * (p_vec[i] as f64)) / (a_vec[i] as f64)) as f32;
    }
    Tensor::from_vec(out, d, param.device())
}

/// Iteratively Reweighted Least Squares (IRLS) step.
pub fn step_irls(param: &Tensor, target: &Tensor, mu: f64, eps: f64) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let t_vec = target.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let r = ((p_vec[i] as f64) - (t_vec[i] as f64)).abs();
        let w = 1.0 / (r + eps);
        let tgt = t_vec[i] as f64;
        let p = p_vec[i] as f64;
        out[i] = ((w * tgt + mu * p) / (w + mu)) as f32;
    }
    Tensor::from_vec(out, d, param.device())
}

/// Concave-Convex Procedure (CCCP) majorization step: (2 * x) / (1 + x^2).
pub fn step_cccp(param: &Tensor) -> Result<Tensor> {
    let p_vec = param.to_vec1::<f32>()?;
    let d = p_vec.len();

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let x = p_vec[i] as f64;
        out[i] = ((2.0 * x) / (1.0 + x * x)) as f32;
    }
    Tensor::from_vec(out, d, param.device())
}

/// Unified module encapsulating coordinate and MM optimization for parity verification.
pub struct CoordinateMMUnifiedModule {
    pub x0: Tensor,
    pub normal: Tensor,
    pub offset: f64,
    pub param_bcd: Tensor,
    pub b_bcd: Tensor,
    pub a_diag: Tensor,
    pub param_irls: Tensor,
    pub target_irls: Tensor,
    pub param_cccp: Tensor,
    pub anchor: LinearAnchor,
    pub golden_dykstra_x: Tensor,
    pub golden_dykstra_y: Tensor,
    pub golden_dykstra_p: Tensor,
    pub golden_dykstra_q: Tensor,
    pub golden_param_bcd: Tensor,
    pub golden_param_irls: Tensor,
    pub golden_param_cccp: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl CoordinateMMUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let x0 = tensors
            .get("x0")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing x0".into()))?;
        let normal = tensors
            .get("normal")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing normal".into()))?;
        let offset = tensors
            .get("offset")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing offset".into()))?
            .to_vec1::<f32>()?[0] as f64;

        let param_bcd = tensors
            .get("param_bcd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_bcd".into()))?;
        let b_bcd = tensors
            .get("b_bcd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing b_bcd".into()))?;
        let a_diag = tensors
            .get("A_diag")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing A_diag".into()))?;

        let param_irls = tensors
            .get("param_irls")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_irls".into()))?;
        let target_irls = tensors
            .get("target_irls")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing target_irls".into()))?;

        let param_cccp = tensors
            .get("param_cccp")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing param_cccp".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_dykstra_x = tensors
            .get("golden_dykstra_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_dykstra_x".into()))?;
        let golden_dykstra_y = tensors
            .get("golden_dykstra_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_dykstra_y".into()))?;
        let golden_dykstra_p = tensors
            .get("golden_dykstra_p")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_dykstra_p".into()))?;
        let golden_dykstra_q = tensors
            .get("golden_dykstra_q")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_dykstra_q".into()))?;
        let golden_param_bcd = tensors
            .get("golden_param_bcd")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_bcd".into()))?;
        let golden_param_irls = tensors
            .get("golden_param_irls")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_irls".into()))?;
        let golden_param_cccp = tensors
            .get("golden_param_cccp")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_param_cccp".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            x0,
            normal,
            offset,
            param_bcd,
            b_bcd,
            a_diag,
            param_irls,
            target_irls,
            param_cccp,
            anchor,
            golden_dykstra_x,
            golden_dykstra_y,
            golden_dykstra_p,
            golden_dykstra_q,
            golden_param_bcd,
            golden_param_irls,
            golden_param_cccp,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: Dykstra's Alternating Projections
        let p0 = self.x0.zeros_like()?;
        let q0 = self.x0.zeros_like()?;
        let (x1, y1, p1, q1) =
            step_dykstra(&self.x0, &p0, &q0, &self.normal, self.offset, 0.0, 2.0)?;
        let err_x = max_abs_diff(&x1, &self.golden_dykstra_x)?;
        let err_y = max_abs_diff(&y1, &self.golden_dykstra_y)?;
        let err_p = max_abs_diff(&p1, &self.golden_dykstra_p)?;
        let err_q = max_abs_diff(&q1, &self.golden_dykstra_q)?;
        let err_dykstra = err_x.max(err_y).max(err_p).max(err_q);
        if err_dykstra >= atol {
            candle_core::bail!(
                "Dykstra invariant failed: err={:e} >= {:e}",
                err_dykstra,
                atol
            );
        }
        errors.insert("dykstra_err".to_string(), err_dykstra);

        // Invariant 2: Block Coordinate Descent (BCD)
        let p_bcd = step_bcd(&self.param_bcd, &self.b_bcd, &self.a_diag, 0.2)?;
        let err_bcd = max_abs_diff(&p_bcd, &self.golden_param_bcd)?;
        if err_bcd >= atol {
            candle_core::bail!("BCD invariant failed: err={:e} >= {:e}", err_bcd, atol);
        }
        errors.insert("bcd_err".to_string(), err_bcd);

        // Invariant 3: Iteratively Reweighted Least Squares (IRLS)
        let p_irls = step_irls(&self.param_irls, &self.target_irls, 0.5, 1e-4)?;
        let err_irls = max_abs_diff(&p_irls, &self.golden_param_irls)?;
        if err_irls >= atol {
            candle_core::bail!("IRLS invariant failed: err={:e} >= {:e}", err_irls, atol);
        }
        errors.insert("irls_err".to_string(), err_irls);

        // Invariant 4: Concave-Convex Procedure (CCCP)
        let p_cccp = step_cccp(&self.param_cccp)?;
        let err_cccp = max_abs_diff(&p_cccp, &self.golden_param_cccp)?;
        if err_cccp >= atol {
            candle_core::bail!("CCCP invariant failed: err={:e} >= {:e}", err_cccp, atol);
        }
        errors.insert("cccp_err".to_string(), err_cccp);

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
