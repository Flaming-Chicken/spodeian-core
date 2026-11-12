//! Surrogate Models Bayesian Optimization & Differentiable Proxies
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Invert an N x N matrix on CPU with f64 precision.
fn invert_matrix(mat: &Tensor) -> Result<Tensor> {
    let (n, n2) = mat.dims2()?;
    assert_eq!(n, n2);
    let flat = mat.to_vec2::<f32>()?;
    let mut aug = vec![vec![0f64; 2 * n]; n];
    for r in 0..n {
        for c in 0..n {
            aug[r][c] = flat[r][c] as f64;
        }
        aug[r][n + r] = 1.0f64;
    }
    for i in 0..n {
        // Partial pivoting: find max pivot in column i
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for r in (i + 1)..n {
            if aug[r][i].abs() > max_val {
                max_val = aug[r][i].abs();
                max_row = r;
            }
        }
        if max_row != i {
            aug.swap(i, max_row);
        }
        let pivot = aug[i][i];
        assert!(pivot.abs() > 1e-12, "Singular matrix in inversion");
        for c in 0..(2 * n) {
            aug[i][c] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = aug[r][i];
                for c in 0..(2 * n) {
                    aug[r][c] -= factor * aug[i][c];
                }
            }
        }
    }
    let mut inv_flat = vec![0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            inv_flat[r * n + c] = aug[r][n + c] as f32;
        }
    }
    Tensor::from_vec(inv_flat, (n, n), mat.device())
}

/// RBF (Squared Exponential) Kernel:
/// K(x1, x2) = variance * exp( -0.5 * sum_d ((x1 - x2)/lengthscale)^2 )
pub fn rbf_kernel(
    x1: &Tensor,
    x2: &Tensor,
    lengthscale: &Tensor,
    variance: &Tensor,
) -> Result<Tensor> {
    let (n1, d) = x1.dims2()?;
    let (n2, d2) = x2.dims2()?;
    assert_eq!(d, d2);

    let x1_vec = x1.to_vec2::<f32>()?;
    let x2_vec = x2.to_vec2::<f32>()?;
    let l_vec = lengthscale.to_vec1::<f32>()?;
    let var_scalar: f32 = variance.to_scalar()?;

    let mut k_flat = vec![0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let mut dist_sq = 0.0f32;
            for col in 0..d {
                let diff = (x1_vec[i][col] - x2_vec[j][col]) / l_vec[col];
                dist_sq += diff * diff;
            }
            k_flat[i * n2 + j] = var_scalar * (-0.5f32 * dist_sq).exp();
        }
    }
    Tensor::from_vec(k_flat, (n1, n2), x1.device())
}

/// Gaussian Process Posterior Mean and Variance
pub fn gp_posterior(
    x_train: &Tensor,
    y_train: &Tensor,
    x_test: &Tensor,
    lengthscale: &Tensor,
    variance: &Tensor,
    noise_var: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (n_train, _) = x_train.dims2()?;
    let (n_test, _) = x_test.dims2()?;

    let k_yy_base = rbf_kernel(x_train, x_train, lengthscale, variance)?;
    let noise_scalar: f32 = noise_var.to_scalar()?;
    let eye_n = Tensor::eye(n_train, candle_core::DType::F32, x_train.device())?;
    let noise_eye = eye_n.affine(noise_scalar as f64, 0.0)?;
    let k_yy = k_yy_base.add(&noise_eye)?;

    let k_sy = rbf_kernel(x_test, x_train, lengthscale, variance)?;
    let k_ss = rbf_kernel(x_test, x_test, lengthscale, variance)?;

    let k_yy_inv = invert_matrix(&k_yy)?;

    // alpha = K_yy^{-1} y
    let y_col = y_train.reshape((n_train, 1))?;
    let alpha = k_yy_inv.matmul(&y_col)?;
    let mu_s = k_sy.matmul(&alpha)?.reshape((n_test,))?;

    // V = K_yy^{-1} K_sy^T
    let k_sy_t = k_sy.t()?;
    let v = k_yy_inv.matmul(&k_sy_t)?;
    let k_sy_v = k_sy.matmul(&v)?;
    let cov_s = k_ss.sub(&k_sy_v)?;

    // Extract diagonal
    let cov_vec = cov_s.to_vec2::<f32>()?;
    let mut var_diag = vec![0f32; n_test];
    for i in 0..n_test {
        var_diag[i] = cov_vec[i][i].max(1e-6f32);
    }
    let var_s = Tensor::from_vec(var_diag, (n_test,), x_test.device())?;

    Ok((mu_s, var_s))
}

/// Expected Improvement (EI) acquisition function using Abramowitz-Stegun erf
pub fn expected_improvement(mu: &Tensor, var: &Tensor, best_y: &Tensor, xi: f64) -> Result<Tensor> {
    let n = mu.dims1()?;
    let mu_vec = mu.to_vec1::<f32>()?;
    let var_vec = var.to_vec1::<f32>()?;
    let best_scalar: f32 = best_y.to_scalar()?;

    let a1: f64 = 0.254829592;
    let a2: f64 = -0.284496736;
    let a3: f64 = 1.421413741;
    let a4: f64 = -1.453152027;
    let a5: f64 = 1.061405429;
    let p_coeff: f64 = 0.3275911;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    let mut ei_vec = vec![0f32; n];
    for i in 0..n {
        let sigma = (var_vec[i] as f64).sqrt();
        let improvement = (mu_vec[i] as f64) - (best_scalar as f64) - xi;
        let z = improvement / sigma;

        let phi_z = inv_sqrt_2pi * (-0.5 * z * z).exp();

        // Erf(z / sqrt(2)) via Abramowitz-Stegun formula 7.1.26
        let x_arg = z / std::f64::consts::SQRT_2;
        let sign = if x_arg >= 0.0 { 1.0 } else { -1.0 };
        let abs_x = x_arg.abs();
        let t = 1.0 / (1.0 + p_coeff * abs_x);
        let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        let erf_val = sign * (1.0 - poly * (-abs_x * abs_x).exp());
        let phi_cdf = 0.5 * (1.0 + erf_val);

        let val = improvement * phi_cdf + sigma * phi_z;
        ei_vec[i] = (val.max(0.0)) as f32;
    }
    Tensor::from_vec(ei_vec, (n,), mu.device())
}

/// Upper Confidence Bound (UCB) acquisition function: mu + beta * sqrt(var)
pub fn ucb_acquisition(mu: &Tensor, var: &Tensor, beta: f64) -> Result<Tensor> {
    let sigma = var.sqrt()?;
    let beta_sigma = sigma.affine(beta, 0.0)?;
    mu.add(&beta_sigma)
}
