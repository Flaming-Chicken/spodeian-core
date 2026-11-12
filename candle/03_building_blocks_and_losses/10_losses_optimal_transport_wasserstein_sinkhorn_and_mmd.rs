//! Losses: Optimal Transport, Wasserstein, Sinkhorn, and MMD (C03 Topic 10).
//!
//! Native Rust Candle implementation of:
//! 1. Exact 1D Wasserstein Distance via sorted order statistics.
//! 2. Sliced Wasserstein Distance (SWD) via random 1D projections.
//! 3. Entropic Regularized Optimal Transport (Sinkhorn Distance).
//! 4. Debiased Sinkhorn Divergence with self-transport correction.
//! 5. Maximum Mean Discrepancy (MMD^2) with multi-scale Gaussian RBF kernels.
//! 6. Energy / Cramer Distance.
//! 7. Linear Anchor.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// Exact 1D Wasserstein Distance: mean |u_(i) - v_(i)| along sorted order statistics.
pub fn wasserstein_1d_forward(u: &Tensor, v: &Tensor) -> Result<Tensor> {
    let mut u_vec = u.to_vec2::<f32>()?;
    let mut v_vec = v.to_vec2::<f32>()?;
    let b = u_vec.len();
    let n = u_vec[0].len();

    let mut total = 0.0f32;
    for i in 0..b {
        u_vec[i].sort_by(|a, b| a.total_cmp(b));
        v_vec[i].sort_by(|a, b| a.total_cmp(b));
        for j in 0..n {
            total += (u_vec[i][j] - v_vec[i][j]).abs();
        }
    }
    let mean = total / ((b * n) as f32);
    Tensor::new(&[mean], u.device())
}

/// Sliced Wasserstein Distance: average 1D Wasserstein distance across 1D projections.
pub fn sliced_wasserstein_forward(x: &Tensor, y: &Tensor, projections: &Tensor) -> Result<Tensor> {
    let (b, n, d) = x.dims3()?;
    let (l, _) = projections.dims2()?;
    let p_t = projections.t()?;

    let x_flat = x.reshape((b * n, d))?;
    let y_flat = y.reshape((b * n, d))?;
    let proj_x = x_flat.matmul(&p_t)?.reshape((b, n, l))?;
    let proj_y = y_flat.matmul(&p_t)?.reshape((b, n, l))?;
    let proj_x = proj_x.transpose(1, 2)?; // [B, L, N]
    let proj_y = proj_y.transpose(1, 2)?;

    let mut x_vec = proj_x.to_vec3::<f32>()?;
    let mut y_vec = proj_y.to_vec3::<f32>()?;

    let mut total = 0.0f32;
    for b_idx in 0..b {
        for l_idx in 0..l {
            x_vec[b_idx][l_idx].sort_by(|a, b| a.total_cmp(b));
            y_vec[b_idx][l_idx].sort_by(|a, b| a.total_cmp(b));
            for i in 0..n {
                total += (x_vec[b_idx][l_idx][i] - y_vec[b_idx][l_idx][i]).abs();
            }
        }
    }
    let mean = total / ((b * l * n) as f32);
    Tensor::new(&[mean], x.device())
}

/// Helper: compute pairwise squared Euclidean distance matrix C_ij = ||x_i - y_j||^2.
pub fn compute_pairwise_sqdist(x: &Tensor, y: &Tensor) -> Result<Vec<Vec<f32>>> {
    let x_vec = x.to_vec2::<f32>()?;
    let y_vec = y.to_vec2::<f32>()?;
    let n = x_vec.len();
    let m = y_vec.len();
    let d = x_vec[0].len();

    let mut c = vec![vec![0.0f32; m]; n];
    for i in 0..n {
        for j in 0..m {
            let mut sum_sq = 0.0f32;
            for k in 0..d {
                let diff = x_vec[i][k] - y_vec[j][k];
                sum_sq += diff * diff;
            }
            c[i][j] = sum_sq;
        }
    }
    Ok(c)
}

/// Entropic Regularized Optimal Transport (Sinkhorn cost) under squared Euclidean cost.
pub fn sinkhorn_cost_forward(x: &Tensor, y: &Tensor, eps: f32, num_iters: usize) -> Result<Tensor> {
    let c = compute_pairwise_sqdist(x, y)?;
    let n = c.len();
    let m = c[0].len();

    let mut k = vec![vec![0.0f32; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = (-c[i][j] / eps).exp();
        }
    }

    let a_val = 1.0f32 / (n as f32);
    let b_val = 1.0f32 / (m as f32);
    let mut u = vec![a_val; n];
    let mut v = vec![b_val; m];

    for _ in 0..num_iters {
        // u = a / (K * v + 1e-12)
        for i in 0..n {
            let mut kv_i = 0.0f32;
            for j in 0..m {
                kv_i += k[i][j] * v[j];
            }
            u[i] = a_val / (kv_i + 1e-12);
        }

        // v = b / (K^T * u + 1e-12)
        for j in 0..m {
            let mut ktu_j = 0.0f32;
            for i in 0..n {
                ktu_j += k[i][j] * u[i];
            }
            v[j] = b_val / (ktu_j + 1e-12);
        }
    }

    // Transport cost = sum_ij u_i * K_ij * v_j * C_ij
    let mut cost = 0.0f32;
    for i in 0..n {
        for j in 0..m {
            let p_ij = u[i] * k[i][j] * v[j];
            cost += p_ij * c[i][j];
        }
    }

    Tensor::new(&[cost], x.device())
}

/// Debiased Sinkhorn Divergence: S_eps(X, Y) - 0.5 * S_eps(X, X) - 0.5 * S_eps(Y, Y).
pub fn debiased_sinkhorn_forward(
    x: &Tensor,
    y: &Tensor,
    eps: f32,
    num_iters: usize,
) -> Result<Tensor> {
    let s_xy = sinkhorn_cost_forward(x, y, eps, num_iters)?;
    let s_xx = sinkhorn_cost_forward(x, x, eps, num_iters)?;
    let s_yy = sinkhorn_cost_forward(y, y, eps, num_iters)?;

    let s_xy_val = s_xy.to_vec1::<f32>()?[0];
    let s_xx_val = s_xx.to_vec1::<f32>()?[0];
    let s_yy_val = s_yy.to_vec1::<f32>()?[0];

    let debiased = s_xy_val - 0.5f32 * s_xx_val - 0.5f32 * s_yy_val;
    Tensor::new(&[debiased], x.device())
}

/// Maximum Mean Discrepancy (MMD^2) with multi-scale Gaussian RBF kernels.
pub fn mmd_forward(x: &Tensor, y: &Tensor, sigmas: &[f32]) -> Result<Tensor> {
    let c_xx = compute_pairwise_sqdist(x, x)?;
    let c_yy = compute_pairwise_sqdist(y, y)?;
    let c_xy = compute_pairwise_sqdist(x, y)?;

    let compute_k_mean = |c_mat: &[Vec<f32>]| -> f32 {
        let rows = c_mat.len();
        let cols = c_mat[0].len();
        let mut sum_k = 0.0f32;
        for i in 0..rows {
            for j in 0..cols {
                let d2 = c_mat[i][j];
                for &s in sigmas {
                    let gamma = 1.0f32 / (2.0f32 * s * s);
                    sum_k += (-gamma * d2).exp();
                }
            }
        }
        sum_k / ((rows * cols) as f32)
    };

    let mean_xx = compute_k_mean(&c_xx);
    let mean_yy = compute_k_mean(&c_yy);
    let mean_xy = compute_k_mean(&c_xy);

    let mmd2 = mean_xx + mean_yy - 2.0f32 * mean_xy;
    Tensor::new(&[mmd2], x.device())
}

/// Energy Distance / Cramer Distance: 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||.
pub fn energy_distance_forward(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let y_vec = y.to_vec2::<f32>()?;
    let compute_dist_mean = |a: &[Vec<f32>], b: &[Vec<f32>]| -> f32 {
        let rows = a.len();
        let cols = b.len();
        let d = a[0].len();
        let mut sum_dist = 0.0f32;
        for i in 0..rows {
            for j in 0..cols {
                let mut sum_sq = 0.0f32;
                for k in 0..d {
                    let diff = a[i][k] - b[j][k];
                    sum_sq += diff * diff;
                }
                sum_dist += sum_sq.sqrt();
            }
        }
        sum_dist / ((rows * cols) as f32)
    };

    let mean_xy = compute_dist_mean(&x_vec, &y_vec);
    let mean_xx = compute_dist_mean(&x_vec, &x_vec);
    let mean_yy = compute_dist_mean(&y_vec, &y_vec);

    let energy = 2.0f32 * mean_xy - mean_xx - mean_yy;
    Tensor::new(&[energy], x.device())
}
