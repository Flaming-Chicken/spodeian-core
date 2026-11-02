//! Support Vector Machines and Kernel Methods in Native Rust Candle.
//!
//! Provides:
//! - Pairwise Euclidean squared distance and RBF Gram matrix computation
//! - Polynomial kernel Gram matrix computation
//! - Kernel Ridge Regression dual solver via pure-Rust Cholesky decomposition
//! - Nyström low-rank Gram matrix landmark approximation

use candle_core::{Result, Tensor};

/// Computes pairwise squared Euclidean distances: D_ij = ||x_i - x_j||^2.
pub fn pairwise_sq_dist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let (n1, _d) = x1.dims2()?;
    let (n2, _) = x2.dims2()?;

    let x1_sq = x1.sqr()?.sum_keepdim(1)?; // [n1, 1]
    let x2_sq = x2.sqr()?.sum_keepdim(1)?.t()?; // [1, n2]

    let dot = x1.matmul(&x2.t()?)?; // [n1, n2]
    let dist_sq =
        ((x1_sq.broadcast_as((n1, n2))? + x2_sq.broadcast_as((n1, n2))?)? - (dot * 2.0)?)?;
    dist_sq.clamp(0.0f32, f32::MAX)
}

/// Computes RBF Gaussian kernel Gram matrix: K_ij = exp(-gamma * ||x_i - x_j||^2).
pub fn rbf_kernel(x1: &Tensor, x2: &Tensor, gamma: f64) -> Result<Tensor> {
    let dist_sq = pairwise_sq_dist(x1, x2)?;
    (dist_sq * (-gamma))?.exp()
}

/// Computes Polynomial kernel Gram matrix: K_ij = (x_i^T x_j / D + coef0)^degree.
pub fn poly_kernel(x1: &Tensor, x2: &Tensor, degree: f64, coef0: f64) -> Result<Tensor> {
    let (_n1, d) = x1.dims2()?;
    let dot = x1.matmul(&x2.t()?)?;
    let scaled = (dot / (d as f64))?;
    let shifted = (scaled + coef0)?;
    shifted.powf(degree)
}

/// Solves symmetric positive-definite linear system A x = b via Cholesky decomposition L L^T = A.
pub fn solve_spd_cholesky(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (n, n_cols) = a.dims2()?;
    if n != n_cols {
        candle_core::bail!("Matrix A must be square, got ({n}, {n_cols})");
    }

    let a_vec: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let b_vec: Vec<f32> = b.flatten_all()?.to_vec1()?;

    let mut l = vec![0.0f32; n * n];

    // Cholesky decomposition: A = L L^T
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a_vec[i * n + i] - sum;
                l[i * n + j] = if diag > 1e-12 { diag.sqrt() } else { 1e-6 };
            } else {
                let diag_j = l[j * n + j];
                l[i * n + j] = if diag_j.abs() > 1e-12 {
                    (a_vec[i * n + j] - sum) / diag_j
                } else {
                    0.0
                };
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0f32;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-12 {
            (b_vec[i] - sum) / diag
        } else {
            0.0
        };
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = 0.0f32;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-12 {
            (y[i] - sum) / diag
        } else {
            0.0
        };
    }

    Tensor::from_vec(x, (n,), a.device())
}

/// Solves Kernel Ridge Regression: (K + alpha * I) c = y.
pub fn solve_kernel_ridge(x: &Tensor, y: &Tensor, gamma: f64, alpha: f64) -> Result<Tensor> {
    let (n, _) = x.dims2()?;
    let k = rbf_kernel(x, x, gamma)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, x.device())?;
    let reg_k = (k + (eye * alpha)?)?;
    solve_spd_cholesky(&reg_k, y)
}

/// Computes Nyström low-rank Gram approximation: K_nm @ pinv(K_mm) @ K_nm.T.
pub fn nystrom_approximate_gram(
    x: &Tensor,
    landmarks: &Tensor,
    gamma: f64,
    eps: f64,
) -> Result<Tensor> {
    let (m, _) = landmarks.dims2()?;
    let k_mm = rbf_kernel(landmarks, landmarks, gamma)?;
    let k_nm = rbf_kernel(x, landmarks, gamma)?;
    let eye_m = Tensor::eye(m, candle_core::DType::F32, x.device())?;
    let reg_k_mm = (k_mm + (eye_m * eps)?)?;

    // Invert reg_k_mm via Cholesky column-by-column
    let mut inv_cols = Vec::with_capacity(m);
    for col in 0..m {
        let mut e = vec![0.0f32; m];
        e[col] = 1.0;
        let e_t = Tensor::from_vec(e, (m,), x.device())?;
        let col_sol = solve_spd_cholesky(&reg_k_mm, &e_t)?;
        inv_cols.push(col_sol.unsqueeze(1)?);
    }
    let k_mm_inv = Tensor::cat(&inv_cols, 1)?; // [m, m]

    let temp = k_nm.matmul(&k_mm_inv)?; // [n, m]
    temp.matmul(&k_nm.t()?) // [n, n]
}
