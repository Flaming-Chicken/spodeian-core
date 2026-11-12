//! Candle Implementation for Category 04 Topic 14 (Subfolder 02 Topic 07):
//! Minimax Saddle Point, Extragradient, OGDA, SGA, and Trust-Region BOBYQA.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Korpelevich Extragradient step on bilinear and minimax games.
//! 2. Optimistic Gradient Descent-Ascent (OGDA) with negative momentum.
//! 3. Symplectic Gradient Adjustment (SGA) Hamiltonian vector field alignment.
//! 4. Nelder-Mead derivative-free simplex reflection dynamics.

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

/// Helper to multiply 2x2 matrix by 2x1 vector.
fn mat_vec_mul(a: &[f32], v: &[f32]) -> [f32; 2] {
    [a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1]]
}

/// Helper to multiply 2x2 matrix transpose by 2x1 vector.
fn mat_t_vec_mul(a: &[f32], v: &[f32]) -> [f32; 2] {
    [a[0] * v[0] + a[2] * v[1], a[1] * v[0] + a[3] * v[1]]
}

/// Korpelevich Extragradient step on bilinear game min_x max_y x^T A y.
pub fn step_extragradient(
    x: &Tensor,
    y: &Tensor,
    a: &Tensor,
    eta_x: f64,
    eta_y: f64,
) -> Result<(Tensor, Tensor)> {
    let x_v = x.to_vec1::<f32>()?;
    let y_v = y.to_vec1::<f32>()?;
    let a_flat = a.flatten_all()?.to_vec1::<f32>()?;

    let gx = mat_vec_mul(&a_flat, &y_v);
    let gy = mat_t_vec_mul(&a_flat, &x_v);

    let x_tilde = [
        x_v[0] - (eta_x as f32) * gx[0],
        x_v[1] - (eta_x as f32) * gx[1],
    ];
    let y_tilde = [
        y_v[0] + (eta_y as f32) * gy[0],
        y_v[1] + (eta_y as f32) * gy[1],
    ];

    let gx_tilde = mat_vec_mul(&a_flat, &y_tilde);
    let gy_tilde = mat_t_vec_mul(&a_flat, &x_tilde);

    let x_next = [
        x_v[0] - (eta_x as f32) * gx_tilde[0],
        x_v[1] - (eta_x as f32) * gx_tilde[1],
    ];
    let y_next = [
        y_v[0] + (eta_y as f32) * gy_tilde[0],
        y_v[1] + (eta_y as f32) * gy_tilde[1],
    ];

    let dev = x.device();
    Ok((
        Tensor::from_vec(x_next.to_vec(), 2, dev)?,
        Tensor::from_vec(y_next.to_vec(), 2, dev)?,
    ))
}

/// Optimistic Gradient Descent-Ascent (OGDA) step.
pub fn step_ogda(
    x: &Tensor,
    y: &Tensor,
    x_prev: &Tensor,
    y_prev: &Tensor,
    a: &Tensor,
    eta_x: f64,
    eta_y: f64,
) -> Result<(Tensor, Tensor)> {
    let x_v = x.to_vec1::<f32>()?;
    let y_v = y.to_vec1::<f32>()?;
    let xp_v = x_prev.to_vec1::<f32>()?;
    let yp_v = y_prev.to_vec1::<f32>()?;
    let a_flat = a.flatten_all()?.to_vec1::<f32>()?;

    let gx = mat_vec_mul(&a_flat, &y_v);
    let gy = mat_t_vec_mul(&a_flat, &x_v);
    let gxp = mat_vec_mul(&a_flat, &yp_v);
    let gyp = mat_t_vec_mul(&a_flat, &xp_v);

    let ex = eta_x as f32;
    let ey = eta_y as f32;

    let x_next = [
        x_v[0] - 2.0 * ex * gx[0] + ex * gxp[0],
        x_v[1] - 2.0 * ex * gx[1] + ex * gxp[1],
    ];
    let y_next = [
        y_v[0] + 2.0 * ey * gy[0] - ey * gyp[0],
        y_v[1] + 2.0 * ey * gy[1] - ey * gyp[1],
    ];

    let dev = x.device();
    Ok((
        Tensor::from_vec(x_next.to_vec(), 2, dev)?,
        Tensor::from_vec(y_next.to_vec(), 2, dev)?,
    ))
}

/// Symplectic Gradient Adjustment (SGA) step.
pub fn step_sga(
    x: &Tensor,
    y: &Tensor,
    a: &Tensor,
    lambda_sga: f64,
    eta_x: f64,
    eta_y: f64,
) -> Result<(Tensor, Tensor)> {
    let x_v = x.to_vec1::<f32>()?;
    let y_v = y.to_vec1::<f32>()?;
    let a_flat = a.flatten_all()?.to_vec1::<f32>()?;

    let at_x = mat_t_vec_mul(&a_flat, &x_v);
    let a_at_x = mat_vec_mul(&a_flat, &at_x);

    let a_y = mat_vec_mul(&a_flat, &y_v);
    let at_a_y = mat_t_vec_mul(&a_flat, &a_y);

    let gx = mat_vec_mul(&a_flat, &y_v);
    let gy = mat_t_vec_mul(&a_flat, &x_v);

    let lam = lambda_sga as f32;
    let ex = eta_x as f32;
    let ey = eta_y as f32;

    let ux = [gx[0] + lam * a_at_x[0], gx[1] + lam * a_at_x[1]];
    let uy = [gy[0] - lam * at_a_y[0], gy[1] - lam * at_a_y[1]];

    let x_next = [x_v[0] - ex * ux[0], x_v[1] - ex * ux[1]];
    let y_next = [y_v[0] + ey * uy[0], y_v[1] + ey * uy[1]];

    let dev = x.device();
    Ok((
        Tensor::from_vec(x_next.to_vec(), 2, dev)?,
        Tensor::from_vec(y_next.to_vec(), 2, dev)?,
    ))
}

/// Nelder-Mead reflection point calculation.
pub fn nelder_mead_reflect(
    x_best: &Tensor,
    x_second: &Tensor,
    x_worst: &Tensor,
    alpha: f64,
) -> Result<Tensor> {
    let b = x_best.to_vec1::<f32>()?;
    let s = x_second.to_vec1::<f32>()?;
    let w = x_worst.to_vec1::<f32>()?;
    let n = b.len();

    let a = alpha as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let centroid = 0.5 * (b[i] + s[i]);
        out[i] = centroid + a * (centroid - w[i]);
    }

    Tensor::from_vec(out, n, x_best.device())
}

/// Unified module encapsulating minimax, extragradient, and derivative-free algorithms.
pub struct MinimaxUnifiedModule {
    pub matrix_a: Tensor,
    pub state_x: Tensor,
    pub state_y: Tensor,
    pub state_x_prev: Tensor,
    pub state_y_prev: Tensor,

    pub golden_x_eg: Tensor,
    pub golden_y_eg: Tensor,
    pub golden_x_ogda: Tensor,
    pub golden_y_ogda: Tensor,
    pub golden_x_sga: Tensor,
    pub golden_y_sga: Tensor,

    pub x_best: Tensor,
    pub x_second: Tensor,
    pub x_worst: Tensor,
    pub golden_x_reflect: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl MinimaxUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let matrix_a = tensors
            .get("matrix_A")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing matrix_A".into()))?;
        let state_x = tensors
            .get("state_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing state_x".into()))?;
        let state_y = tensors
            .get("state_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing state_y".into()))?;
        let state_x_prev = tensors
            .get("state_x_prev")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing state_x_prev".into()))?;
        let state_y_prev = tensors
            .get("state_y_prev")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing state_y_prev".into()))?;

        let golden_x_eg = tensors
            .get("golden_x_eg")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_x_eg".into()))?;
        let golden_y_eg = tensors
            .get("golden_y_eg")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_y_eg".into()))?;
        let golden_x_ogda = tensors
            .get("golden_x_ogda")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_x_ogda".into()))?;
        let golden_y_ogda = tensors
            .get("golden_y_ogda")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_y_ogda".into()))?;
        let golden_x_sga = tensors
            .get("golden_x_sga")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_x_sga".into()))?;
        let golden_y_sga = tensors
            .get("golden_y_sga")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_y_sga".into()))?;

        let x_best = tensors
            .get("x_best")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing x_best".into()))?;
        let x_second = tensors
            .get("x_second")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing x_second".into()))?;
        let x_worst = tensors
            .get("x_worst")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing x_worst".into()))?;
        let golden_x_reflect = tensors
            .get("golden_x_reflect")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_x_reflect".into()))?;

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
            matrix_a,
            state_x,
            state_y,
            state_x_prev,
            state_y_prev,
            golden_x_eg,
            golden_y_eg,
            golden_x_ogda,
            golden_y_ogda,
            golden_x_sga,
            golden_y_sga,
            x_best,
            x_second,
            x_worst,
            golden_x_reflect,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // 1. Extragradient Invariant
        let (x_eg, y_eg) =
            step_extragradient(&self.state_x, &self.state_y, &self.matrix_a, 0.1, 0.1)?;
        let err_x_eg = max_abs_diff(&x_eg, &self.golden_x_eg)?;
        let err_y_eg = max_abs_diff(&y_eg, &self.golden_y_eg)?;
        let err_eg = err_x_eg.max(err_y_eg);
        if err_eg >= atol {
            candle_core::bail!(
                "Extragradient invariant failed: err={:e} >= {:e}",
                err_eg,
                atol
            );
        }
        errors.insert("extragradient_err".to_string(), err_eg);

        // 2. OGDA Invariant
        let (x_ogda, y_ogda) = step_ogda(
            &self.state_x,
            &self.state_y,
            &self.state_x_prev,
            &self.state_y_prev,
            &self.matrix_a,
            0.1,
            0.1,
        )?;
        let err_x_ogda = max_abs_diff(&x_ogda, &self.golden_x_ogda)?;
        let err_y_ogda = max_abs_diff(&y_ogda, &self.golden_y_ogda)?;
        let err_ogda = err_x_ogda.max(err_y_ogda);
        if err_ogda >= atol {
            candle_core::bail!("OGDA invariant failed: err={:e} >= {:e}", err_ogda, atol);
        }
        errors.insert("ogda_err".to_string(), err_ogda);

        // 3. SGA Invariant
        let (x_sga, y_sga) = step_sga(&self.state_x, &self.state_y, &self.matrix_a, 0.5, 0.1, 0.1)?;
        let err_x_sga = max_abs_diff(&x_sga, &self.golden_x_sga)?;
        let err_y_sga = max_abs_diff(&y_sga, &self.golden_y_sga)?;
        let err_sga = err_x_sga.max(err_y_sga);
        if err_sga >= atol {
            candle_core::bail!("SGA invariant failed: err={:e} >= {:e}", err_sga, atol);
        }
        errors.insert("sga_err".to_string(), err_sga);

        // 4. Nelder-Mead Reflection Invariant
        let x_refl = nelder_mead_reflect(&self.x_best, &self.x_second, &self.x_worst, 1.0)?;
        let err_refl = max_abs_diff(&x_refl, &self.golden_x_reflect)?;
        if err_refl >= atol {
            candle_core::bail!(
                "Nelder-Mead invariant failed: err={:e} >= {:e}",
                err_refl,
                atol
            );
        }
        errors.insert("nelder_mead_err".to_string(), err_refl);

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
