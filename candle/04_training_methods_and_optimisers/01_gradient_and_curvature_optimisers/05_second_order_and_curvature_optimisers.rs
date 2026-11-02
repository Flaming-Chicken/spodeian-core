//! Candle Implementation for Category 04 Topic 05:
//! Second-Order and Curvature Optimisers (L-BFGS, Sophia, AdaHessian, Levenberg-Marquardt).
//!
//! Provides zero-Python, memory-safe implementations of curvature-informed optimization:
//! 1. L-BFGS Two-Loop Recursion (Limited-memory quasi-Newton)
//! 2. Sophia-G (Second-order Clipped Stochastic Optimization)
//! 3. AdaHessian (Adaptive Hessian Diagonal Scaling)
//! 4. Levenberg-Marquardt (Damped Gauss-Newton Solver with Cholesky Factorization)

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

/// Compute search direction using the classical L-BFGS two-loop recursion.
pub fn lbfgs_two_loop(s_hist: &Tensor, y_hist: &Tensor, grad: &Tensor) -> Result<Tensor> {
    let s = s_hist.to_vec2::<f32>()?;
    let y = y_hist.to_vec2::<f32>()?;
    let g = grad.to_vec1::<f32>()?;
    let m = s.len();
    let d = g.len();

    let mut q = g.clone();
    let mut alphas = vec![0.0f32; m];
    let mut rhos = vec![0.0f32; m];

    // First loop: backward recursion
    for i in (0..m).rev() {
        let mut sy = 0.0f64;
        let mut sq = 0.0f64;
        for j in 0..d {
            sy += (s[i][j] as f64) * (y[i][j] as f64);
            sq += (s[i][j] as f64) * (q[j] as f64);
        }
        let rho_i = (1.0 / sy) as f32;
        rhos[i] = rho_i;
        let alpha_i = (rho_i as f64 * sq) as f32;
        alphas[i] = alpha_i;
        for j in 0..d {
            q[j] -= alpha_i * y[i][j];
        }
    }

    // Initial Hessian scaling: gamma_k * I
    let mut s_last_y = 0.0f64;
    let mut y_last_y = 0.0f64;
    for j in 0..d {
        s_last_y += (s[m - 1][j] as f64) * (y[m - 1][j] as f64);
        y_last_y += (y[m - 1][j] as f64) * (y[m - 1][j] as f64);
    }
    let gamma = (s_last_y / y_last_y) as f32;
    let mut r = vec![0.0f32; d];
    for j in 0..d {
        r[j] = gamma * q[j];
    }

    // Second loop: forward recursion
    for i in 0..m {
        let mut yr = 0.0f64;
        for j in 0..d {
            yr += (y[i][j] as f64) * (r[j] as f64);
        }
        let beta = (rhos[i] as f64 * yr) as f32;
        for j in 0..d {
            r[j] += s[i][j] * (alphas[i] - beta);
        }
    }

    Tensor::from_vec(r, d, grad.device())
}

/// Perform one parameter update step using L-BFGS two-loop search direction.
pub fn step_lbfgs(
    theta: &Tensor,
    s_hist: &Tensor,
    y_hist: &Tensor,
    grad: &Tensor,
    lr: f64,
) -> Result<Tensor> {
    let dir = lbfgs_two_loop(s_hist, y_hist, grad)?;
    let step = (dir * lr)?;
    theta.sub(&step)
}

/// Perform one step of Sophia-G (Second-order Clipped Stochastic Optimization).
pub fn step_sophia(
    theta: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
    hess_diag: &Tensor,
    hess_diag_est: &Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
    rho: f64,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let th = theta.to_vec1::<f32>()?;
    let g = grad.to_vec1::<f32>()?;
    let m = momentum.to_vec1::<f32>()?;
    let h = hess_diag.to_vec1::<f32>()?;
    let h_hat = hess_diag_est.to_vec1::<f32>()?;
    let d = th.len();

    let mut m_next = vec![0.0f32; d];
    let mut h_next = vec![0.0f32; d];
    let mut th_next = vec![0.0f32; d];

    for j in 0..d {
        let m_val = (beta1 * (m[j] as f64) + (1.0 - beta1) * (g[j] as f64)) as f32;
        let h_val = (beta2 * (h[j] as f64) + (1.0 - beta2) * (h_hat[j] as f64)) as f32;
        let denom = (h_val as f64).max(eps);
        let ratio = (m_val as f64) / denom;
        let clipped = ratio.clamp(-rho, rho);
        let th_val = (th[j] as f64) * (1.0 - lr * weight_decay) - lr * clipped;

        m_next[j] = m_val;
        h_next[j] = h_val;
        th_next[j] = th_val as f32;
    }

    let dev = theta.device();
    Ok((
        Tensor::from_vec(th_next, d, dev)?,
        Tensor::from_vec(m_next, d, dev)?,
        Tensor::from_vec(h_next, d, dev)?,
    ))
}

/// Perform one step of AdaHessian (Adaptive Hessian Diagonal Scaling).
pub fn step_adahessian(
    theta: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
    v: &Tensor,
    d_diag: &Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
    step: usize,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let th = theta.to_vec1::<f32>()?;
    let g = grad.to_vec1::<f32>()?;
    let m = momentum.to_vec1::<f32>()?;
    let v_vec = v.to_vec1::<f32>()?;
    let diag = d_diag.to_vec1::<f32>()?;
    let d_dim = th.len();

    let bias_m = 1.0 - beta1.powi(step as i32);
    let bias_v = 1.0 - beta2.powi(step as i32);

    let mut m_next = vec![0.0f32; d_dim];
    let mut v_next = vec![0.0f32; d_dim];
    let mut th_next = vec![0.0f32; d_dim];

    for j in 0..d_dim {
        let m_val = (beta1 * (m[j] as f64) + (1.0 - beta1) * (g[j] as f64)) as f32;
        let d_val = diag[j] as f64;
        let v_val = (beta2 * (v_vec[j] as f64) + (1.0 - beta2) * (d_val * d_val)) as f32;

        let m_hat = (m_val as f64) / bias_m;
        let v_hat = (v_val as f64) / bias_v;

        let update = m_hat / (v_hat.sqrt() + eps);
        let th_val = (th[j] as f64) * (1.0 - lr * weight_decay) - lr * update;

        m_next[j] = m_val;
        v_next[j] = v_val;
        th_next[j] = th_val as f32;
    }

    let dev = theta.device();
    Ok((
        Tensor::from_vec(th_next, d_dim, dev)?,
        Tensor::from_vec(m_next, d_dim, dev)?,
        Tensor::from_vec(v_next, d_dim, dev)?,
    ))
}

/// Cholesky linear solver for symmetric positive definite system A x = b.
fn cholesky_solve(a_mat: &[f64], b_vec: &[f64], n: usize) -> Result<Vec<f32>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a_mat[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    candle_core::bail!("Matrix not positive definite");
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b_vec[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }

    Ok(x.into_iter().map(|v| v as f32).collect())
}

/// Perform one Levenberg-Marquardt damped Gauss-Newton update step.
pub fn step_levenberg_marquardt(
    theta: &Tensor,
    jacobian: &Tensor,
    residual: &Tensor,
    damping: f64,
) -> Result<Tensor> {
    let j_mat = jacobian.to_vec2::<f32>()?;
    let r_vec = residual.to_vec1::<f32>()?;
    let n_res = j_mat.len();
    let d = j_mat[0].len();

    // Compute J^T J + damping * I in f64
    let mut a_mat = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0f64;
            for k in 0..n_res {
                sum += (j_mat[k][i] as f64) * (j_mat[k][j] as f64);
            }
            if i == j {
                sum += damping;
            }
            a_mat[i * d + j] = sum;
        }
    }

    // Compute - J^T R in f64
    let mut b_vec = vec![0.0f64; d];
    for i in 0..d {
        let mut sum = 0.0f64;
        for k in 0..n_res {
            sum += (j_mat[k][i] as f64) * (r_vec[k] as f64);
        }
        b_vec[i] = -sum;
    }

    let delta = cholesky_solve(&a_mat, &b_vec, d)?;
    let delta_t = Tensor::from_vec(delta, d, theta.device())?;
    theta.add(&delta_t)
}

/// Unified module encapsulating second-order and curvature optimization routines in Candle.
pub struct SecondOrderCurvatureUnifiedModule {
    pub theta: Tensor,
    pub grad: Tensor,
    pub momentum: Tensor,
    pub hess_diag: Tensor,
    pub hess_diag_est: Tensor,
    pub ada_v: Tensor,
    pub ada_d: Tensor,
    pub lbfgs_s: Tensor,
    pub lbfgs_y: Tensor,
    pub lm_jacobian: Tensor,
    pub lm_residual: Tensor,
    pub anchor: LinearAnchor,
    pub golden_lbfgs_theta: Tensor,
    pub golden_sophia_theta: Tensor,
    pub golden_adahessian_theta: Tensor,
    pub golden_lm_theta: Tensor,
    pub golden_anchor_out: Tensor,
    pub anchor_x: Tensor,
}

impl SecondOrderCurvatureUnifiedModule {
    pub fn load_from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let data = std::fs::read(path).map_err(candle_core::Error::wrap)?;
        let tensors = candle_core::safetensors::load_buffer(&data, device)?;

        let theta = tensors
            .get("theta")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing theta".into()))?;
        let grad = tensors
            .get("grad")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing grad".into()))?;
        let momentum = tensors
            .get("momentum")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing momentum".into()))?;
        let hess_diag = tensors
            .get("hess_diag")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing hess_diag".into()))?;
        let hess_diag_est = tensors
            .get("hess_diag_est")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing hess_diag_est".into()))?;
        let ada_v = tensors
            .get("ada_v")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing ada_v".into()))?;
        let ada_d = tensors
            .get("ada_d")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing ada_d".into()))?;
        let lbfgs_s = tensors
            .get("lbfgs_s")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing lbfgs_s".into()))?;
        let lbfgs_y = tensors
            .get("lbfgs_y")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing lbfgs_y".into()))?;
        let lm_jacobian = tensors
            .get("lm_jacobian")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing lm_jacobian".into()))?;
        let lm_residual = tensors
            .get("lm_residual")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing lm_residual".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        let golden_lbfgs_theta = tensors
            .get("golden_lbfgs_theta")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_lbfgs_theta".into()))?;
        let golden_sophia_theta = tensors
            .get("golden_sophia_theta")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sophia_theta".into()))?;
        let golden_adahessian_theta = tensors
            .get("golden_adahessian_theta")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_adahessian_theta".into()))?;
        let golden_lm_theta = tensors
            .get("golden_lm_theta")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_lm_theta".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;

        Ok(Self {
            theta,
            grad,
            momentum,
            hess_diag,
            hess_diag_est,
            ada_v,
            ada_d,
            lbfgs_s,
            lbfgs_y,
            lm_jacobian,
            lm_residual,
            anchor,
            golden_lbfgs_theta,
            golden_sophia_theta,
            golden_adahessian_theta,
            golden_lm_theta,
            golden_anchor_out,
            anchor_x,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // Invariant 1: L-BFGS
        let lbfgs_theta = step_lbfgs(&self.theta, &self.lbfgs_s, &self.lbfgs_y, &self.grad, 0.1)?;
        let err_lbfgs = max_abs_diff(&lbfgs_theta, &self.golden_lbfgs_theta)?;
        if err_lbfgs >= atol {
            candle_core::bail!("L-BFGS invariant failed: err={:e} >= {:e}", err_lbfgs, atol);
        }
        errors.insert("lbfgs_err".to_string(), err_lbfgs);

        // Invariant 2: Sophia-G
        let (sophia_theta, _, _) = step_sophia(
            &self.theta,
            &self.grad,
            &self.momentum,
            &self.hess_diag,
            &self.hess_diag_est,
            0.01,
            0.96,
            0.99,
            0.01,
            0.04,
            1e-6,
        )?;
        let err_sophia = max_abs_diff(&sophia_theta, &self.golden_sophia_theta)?;
        if err_sophia >= atol {
            candle_core::bail!(
                "Sophia invariant failed: err={:e} >= {:e}",
                err_sophia,
                atol
            );
        }
        errors.insert("sophia_err".to_string(), err_sophia);

        // Invariant 3: AdaHessian
        let (ada_theta, _, _) = step_adahessian(
            &self.theta,
            &self.grad,
            &self.momentum,
            &self.ada_v,
            &self.ada_d,
            0.01,
            0.9,
            0.999,
            0.01,
            1,
            1e-4,
        )?;
        let err_ada = max_abs_diff(&ada_theta, &self.golden_adahessian_theta)?;
        if err_ada >= atol {
            candle_core::bail!(
                "AdaHessian invariant failed: err={:e} >= {:e}",
                err_ada,
                atol
            );
        }
        errors.insert("adahessian_err".to_string(), err_ada);

        // Invariant 4: Levenberg-Marquardt
        let lm_theta =
            step_levenberg_marquardt(&self.theta, &self.lm_jacobian, &self.lm_residual, 0.1)?;
        let err_lm = max_abs_diff(&lm_theta, &self.golden_lm_theta)?;
        if err_lm >= atol {
            candle_core::bail!("LM invariant failed: err={:e} >= {:e}", err_lm, atol);
        }
        errors.insert("lm_err".to_string(), err_lm);

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
