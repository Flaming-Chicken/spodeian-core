//! Candle Implementation for Category 02 Topic 01:
//! Generalized Linear Models and Proximal Lasso.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Ordinary Least Squares (OLS) via normal equations and Cholesky decomposition.
//! 2. Ridge Regression with L2 regularization.
//! 3. Lasso Regression via FISTA soft-thresholding.
//! 4. Poisson GLM negative log-likelihood gradient step.
//! 5. Robust Huber loss evaluation.
//! 6. Linear anchor evaluation for transitive multi-track parity.

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

/// Helper: Solves A * x = b for symmetric positive-definite A (d x d) via Cholesky decomposition A = L L^T.
fn solve_cholesky_spd(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let val = a[i * d + i] - sum;
                l[i * d + j] = if val > 1e-12f32 { val.sqrt() } else { 1e-6f32 };
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f32; d];
    for i in 0..d {
        let mut sum = 0.0f32;
        for k in 0..i {
            sum += l[i * d + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * d + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0f32; d];
    for i in (0..d).rev() {
        let mut sum = 0.0f32;
        for k in (i + 1)..d {
            sum += l[k * d + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * d + i];
    }

    x
}

/// Solves Ordinary Least Squares via normal equations X^T X w = X^T y.
pub fn solve_ols(x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let (n, d) = x.dims2()?;
    let x_data = x.to_vec2::<f32>()?;
    let y_data = y.flatten_all()?.to_vec1::<f32>()?;

    // Compute X^T X (d x d) and X^T y (d)
    let mut xtx = vec![0.0f32; d * d];
    let mut xty = vec![0.0f32; d];

    for r in 0..n {
        let y_r = y_data[r];
        for i in 0..d {
            xty[i] += x_data[r][i] * y_r;
            for j in 0..d {
                xtx[i * d + j] += x_data[r][i] * x_data[r][j];
            }
        }
    }

    let w_vec = solve_cholesky_spd(&xtx, &xty, d);

    // Compute MSE
    let mut sum_sq = 0.0f32;
    for r in 0..n {
        let mut pred = 0.0f32;
        for j in 0..d {
            pred += x_data[r][j] * w_vec[j];
        }
        let diff = pred - y_data[r];
        sum_sq += diff * diff;
    }
    let mse_val = sum_sq / (n as f32);

    let w_t = Tensor::new(w_vec.as_slice(), dev)?;
    let mse_t = Tensor::new(&[mse_val], dev)?;
    Ok((w_t, mse_t))
}

/// Solves Ridge regression: (X^T X + lambda I) w = X^T y.
pub fn solve_ridge(x: &Tensor, y: &Tensor, lambda_reg: f32) -> Result<Tensor> {
    let dev = x.device();
    let (n, d) = x.dims2()?;
    let x_data = x.to_vec2::<f32>()?;
    let y_data = y.flatten_all()?.to_vec1::<f32>()?;

    let mut xtx = vec![0.0f32; d * d];
    let mut xty = vec![0.0f32; d];

    for r in 0..n {
        let y_r = y_data[r];
        for i in 0..d {
            xty[i] += x_data[r][i] * y_r;
            for j in 0..d {
                xtx[i * d + j] += x_data[r][i] * x_data[r][j];
            }
        }
    }

    for i in 0..d {
        xtx[i * d + i] += lambda_reg;
    }

    let w_vec = solve_cholesky_spd(&xtx, &xty, d);
    Tensor::new(w_vec.as_slice(), dev)
}

/// Solves Lasso regression via FISTA (Beck & Teboulle, 2009).
pub fn solve_lasso_fista(x: &Tensor, y: &Tensor, lambda_reg: f32, steps: usize) -> Result<Tensor> {
    let dev = x.device();
    let (n, d) = x.dims2()?;
    let x_data = x.to_vec2::<f32>()?;
    let y_data = y.flatten_all()?.to_vec1::<f32>()?;

    // Compute max eigenvalue bound of X^T X via Frobenius norm bound: eig_max <= sum |xtx_ij|
    let mut xtx = vec![0.0f32; d * d];
    for r in 0..n {
        for i in 0..d {
            for j in 0..d {
                xtx[i * d + j] += x_data[r][i] * x_data[r][j];
            }
        }
    }

    // Power iteration to find max eigenvalue of X^T X
    let mut v = vec![1.0f32; d];
    for _ in 0..20 {
        let mut next_v = vec![0.0f32; d];
        for i in 0..d {
            for j in 0..d {
                next_v[i] += xtx[i * d + j] * v[j];
            }
        }
        let norm: f32 = next_v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for i in 0..d {
            v[i] = next_v[i] / (norm + 1e-8f32);
        }
    }

    let mut eig_max = 0.0f32;
    for i in 0..d {
        let mut row_sum = 0.0f32;
        for j in 0..d {
            row_sum += xtx[i * d + j] * v[j];
        }
        eig_max += v[i] * row_sum;
    }

    let step_size = 1.0f32 / (eig_max + 1e-4f32);
    let mut w = vec![0.0f32; d];
    let mut y_k = vec![0.0f32; d];
    let mut t_k = 1.0f32;

    for _ in 0..steps {
        // Gradient: X^T (X y_k - y)
        let mut pred = vec![0.0f32; n];
        for r in 0..n {
            for j in 0..d {
                pred[r] += x_data[r][j] * y_k[j];
            }
        }

        let mut grad = vec![0.0f32; d];
        for j in 0..d {
            for r in 0..n {
                grad[j] += x_data[r][j] * (pred[r] - y_data[r]);
            }
        }

        let mut w_next = vec![0.0f32; d];
        let threshold = step_size * lambda_reg;
        for j in 0..d {
            let z = y_k[j] - step_size * grad[j];
            if z > threshold {
                w_next[j] = z - threshold;
            } else if z < -threshold {
                w_next[j] = z + threshold;
            } else {
                w_next[j] = 0.0f32;
            }
        }

        let t_next = (1.0f32 + (1.0f32 + 4.0f32 * t_k * t_k).sqrt()) / 2.0f32;
        let factor = (t_k - 1.0f32) / t_next;
        for j in 0..d {
            y_k[j] = w_next[j] + factor * (w_next[j] - w[j]);
        }
        w = w_next;
        t_k = t_next;
    }

    Tensor::new(w.as_slice(), dev)
}

/// Poisson GLM step with log link: mu = exp(X w), grad = X^T (mu - y).
pub fn step_poisson_glm(
    x: &Tensor,
    y_counts: &Tensor,
    w_init: &Tensor,
    lr: f32,
) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let (n, d) = x.dims2()?;
    let x_data = x.to_vec2::<f32>()?;
    let y_vec = y_counts.flatten_all()?.to_vec1::<f32>()?;
    let w_vec = w_init.flatten_all()?.to_vec1::<f32>()?;

    let mut mu = vec![0.0f32; n];
    for r in 0..n {
        let mut lin = 0.0f32;
        for j in 0..d {
            lin += x_data[r][j] * w_vec[j];
        }
        mu[r] = lin.exp();
    }

    let mut grad = vec![0.0f32; d];
    for j in 0..d {
        for r in 0..n {
            grad[j] += x_data[r][j] * (mu[r] - y_vec[r]);
        }
    }

    let mut w_next = vec![0.0f32; d];
    for j in 0..d {
        w_next[j] = w_vec[j] - lr * grad[j];
    }

    let g_t = Tensor::new(grad.as_slice(), dev)?;
    let w_t = Tensor::new(w_next.as_slice(), dev)?;
    Ok((g_t, w_t))
}

/// Mean Huber loss: 0.5 * min(|r|, delta)^2 + delta * (|r| - min(|r|, delta)).
pub fn compute_huber_mean_loss(residual: &Tensor, delta: f32) -> Result<Tensor> {
    let dev = residual.device();
    let r_vec = residual.flatten_all()?.to_vec1::<f32>()?;

    let mut sum_loss = 0.0f32;
    for &r in &r_vec {
        let abs_r = r.abs();
        let quad = abs_r.min(delta);
        let lin = abs_r - quad;
        sum_loss += 0.5f32 * quad * quad + delta * lin;
    }
    let mean_loss = sum_loss / (r_vec.len() as f32);
    Tensor::new(&[mean_loss], dev)
}

/// Unified module for Category 02 Topic 01.
pub struct GLMUnifiedModule {
    pub x_data: Tensor,
    pub y_data: Tensor,
    pub golden_w_ols: Tensor,
    pub golden_mse_ols: Tensor,
    pub golden_w_ridge: Tensor,
    pub golden_w_lasso: Tensor,

    pub w_poisson_init: Tensor,
    pub y_counts: Tensor,
    pub golden_poisson_grad: Tensor,
    pub golden_w_poisson_next: Tensor,
    pub golden_mean_huber: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl GLMUnifiedModule {
    pub fn new(weights_path: &Path) -> Result<Self> {
        let dev = &Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, dev)?;

        let x_data = tensors.get("x_data").unwrap().clone();
        let y_data = tensors.get("y_data").unwrap().clone();
        let golden_w_ols = tensors.get("golden_w_ols").unwrap().clone();
        let golden_mse_ols = tensors.get("golden_mse_ols").unwrap().clone();
        let golden_w_ridge = tensors.get("golden_w_ridge").unwrap().clone();
        let golden_w_lasso = tensors.get("golden_w_lasso").unwrap().clone();

        let w_poisson_init = tensors.get("w_poisson_init").unwrap().clone();
        let y_counts = tensors.get("y_counts").unwrap().clone();
        let golden_poisson_grad = tensors.get("golden_poisson_grad").unwrap().clone();
        let golden_w_poisson_next = tensors.get("golden_w_poisson_next").unwrap().clone();
        let golden_mean_huber = tensors.get("golden_mean_huber").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            x_data,
            y_data,
            golden_w_ols,
            golden_mse_ols,
            golden_w_ridge,
            golden_w_lasso,
            w_poisson_init,
            y_counts,
            golden_poisson_grad,
            golden_w_poisson_next,
            golden_mean_huber,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, tolerance: f32) -> Result<BTreeMap<String, f32>> {
        let mut errs = BTreeMap::new();

        let (w_ols, mse_ols) = solve_ols(&self.x_data, &self.y_data)?;
        let diff_w_ols = w_ols
            .flatten_all()?
            .sub(&self.golden_w_ols.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let diff_mse_ols = (mse_ols.flatten_all()?.to_vec1::<f32>()?[0]
            - self.golden_mse_ols.flatten_all()?.to_vec1::<f32>()?[0])
            .abs();
        let ols_err = diff_w_ols.max(diff_mse_ols);
        errs.insert("ols_err".to_string(), ols_err);

        let w_ridge = solve_ridge(&self.x_data, &self.y_data, 0.5f32)?;
        let ridge_err = w_ridge
            .flatten_all()?
            .sub(&self.golden_w_ridge.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        errs.insert("ridge_err".to_string(), ridge_err);

        let w_lasso = solve_lasso_fista(&self.x_data, &self.y_data, 0.2f32, 10)?;
        let lasso_err = w_lasso
            .flatten_all()?
            .sub(&self.golden_w_lasso.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        errs.insert("lasso_err".to_string(), lasso_err);

        let (p_grad, w_p_next) =
            step_poisson_glm(&self.x_data, &self.y_counts, &self.w_poisson_init, 0.01f32)?;
        let diff_p_grad = p_grad
            .flatten_all()?
            .sub(&self.golden_poisson_grad.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let diff_w_p_next = w_p_next
            .flatten_all()?
            .sub(&self.golden_w_poisson_next.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let poisson_err = diff_p_grad.max(diff_w_p_next);
        errs.insert("poisson_err".to_string(), poisson_err);

        let pred_ols = self.x_data.matmul(&w_ols.unsqueeze(1)?)?.squeeze(1)?;
        let res = pred_ols.sub(&self.y_data)?;
        let huber = compute_huber_mean_loss(&res, 0.5f32)?;
        let huber_err = (huber.flatten_all()?.to_vec1::<f32>()?[0]
            - self.golden_mean_huber.flatten_all()?.to_vec1::<f32>()?[0])
            .abs();
        errs.insert("huber_err".to_string(), huber_err);

        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let diff_anchor = anchor_out
            .sub(&self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        errs.insert("anchor_err".to_string(), diff_anchor);

        for (k, v) in &errs {
            if *v >= tolerance {
                candle_core::bail!(
                    "Candle Category 02 Topic 01 invariant {} error {} >= {}",
                    k,
                    v,
                    tolerance
                );
            }
        }

        Ok(errs)
    }
}
