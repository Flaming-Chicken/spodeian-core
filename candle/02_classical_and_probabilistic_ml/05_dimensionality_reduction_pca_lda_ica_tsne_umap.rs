//! Dimensionality Reduction: PCA, Fisher's LDA, and Kernel PCA in Native Rust Candle.
//!
//! Provides:
//! - Principal Component Analysis via power iteration / SVD projection on mean-centered data
//! - Fisher's Linear Discriminant Analysis for binary class separation via Cholesky solver
//! - Kernel PCA with double-centering of the RBF Gram matrix

use candle_core::{Result, Tensor};

/// Centers matrix X by subtracting column means: X_centered = X - mean(X, axis=0).
pub fn center_matrix(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n, _d) = x.dims2()?;
    let mean = (x.sum_keepdim(0)? / (n as f64))?;
    let centered = (x.broadcast_sub(&mean))?;
    Ok((centered, mean))
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

/// Projects centered data X onto principal axes V_top: proj = (X - mean) @ V_top.
pub fn pca_project(x_centered: &Tensor, v_top: &Tensor) -> Result<Tensor> {
    x_centered.matmul(v_top)
}

/// Solves Fisher's Linear Discriminant direction w_lda = Sw^{-1} (mu1 - mu0).
pub fn solve_lda_direction(x_centered: &Tensor, y: &Tensor, reg: f64) -> Result<Tensor> {
    let (n, d) = x_centered.dims2()?;
    let x_vec = x_centered.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = y.flatten_all()?.to_vec1::<f32>()?;

    let mut n0 = 0.0f32;
    let mut n1 = 0.0f32;
    let mut mu0 = vec![0.0f32; d];
    let mut mu1 = vec![0.0f32; d];

    for i in 0..n {
        if y_vec[i] < 0.5 {
            n0 += 1.0;
            for j in 0..d {
                mu0[j] += x_vec[i * d + j];
            }
        } else {
            n1 += 1.0;
            for j in 0..d {
                mu1[j] += x_vec[i * d + j];
            }
        }
    }

    for j in 0..d {
        mu0[j] /= n0.max(1.0);
        mu1[j] /= n1.max(1.0);
    }

    let mut sw = vec![0.0f32; d * d];
    for i in 0..n {
        let m = if y_vec[i] < 0.5 { &mu0 } else { &mu1 };
        for r in 0..d {
            let diff_r = x_vec[i * d + r] - m[r];
            for c in 0..d {
                let diff_c = x_vec[i * d + c] - m[c];
                sw[r * d + c] += diff_r * diff_c;
            }
        }
    }

    for j in 0..d {
        sw[j * d + j] += reg as f32;
    }

    let sw_tensor = Tensor::from_vec(sw, (d, d), x_centered.device())?;
    let mut diff_mu = vec![0.0f32; d];
    for j in 0..d {
        diff_mu[j] = mu1[j] - mu0[j];
    }
    let diff_mu_tensor = Tensor::from_vec(diff_mu, (d,), x_centered.device())?;

    let w = solve_spd_cholesky(&sw_tensor, &diff_mu_tensor)?;
    let w_norm = w.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    w / (w_norm as f64)
}

/// Double-centers Gram matrix: K_tilde = (I - 1/N 1 1^T) K (I - 1/N 1 1^T).
pub fn double_center_gram(k: &Tensor) -> Result<Tensor> {
    let (n, _) = k.dims2()?;
    let one_n = Tensor::full(1.0f32 / (n as f32), (n, n), k.device())?;

    let left = one_n.matmul(k)?;
    let right = k.matmul(&one_n)?;
    let both = one_n.matmul(&right)?;

    ((k - left)? - right)? + both
}
