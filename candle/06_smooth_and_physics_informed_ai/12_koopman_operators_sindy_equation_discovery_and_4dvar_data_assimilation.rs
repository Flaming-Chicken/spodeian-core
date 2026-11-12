//! Koopman Operators SINDy Equation Discovery & 4D-Var Data Assimilation
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

/// Neural Koopman operator forward step:
/// psi = tanh(x W_enc + b_enc)
/// psi_next = psi @ K_mat
/// x_next = psi_next @ W_dec + b_dec
pub fn koopman_forward(
    x: &Tensor,
    w_enc: &Tensor,
    b_enc: &Tensor,
    k_mat: &Tensor,
    w_dec: &Tensor,
    b_dec: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let lin_enc = x.matmul(w_enc)?.broadcast_add(b_enc)?;
    let psi = lin_enc.tanh()?;
    let psi_next = psi.matmul(k_mat)?;
    let x_next = psi_next.matmul(w_dec)?.broadcast_add(b_dec)?;
    Ok((psi, psi_next, x_next))
}

/// SINDy polynomial + trigonometric candidate library Theta(x) for D=2:
/// [1, x1, x2, x1^2, x1 * x2, x2^2, sin(x1), sin(x2)]
pub fn sindy_library(x: &Tensor) -> Result<Tensor> {
    let (b_sz, _) = x.dims2()?;
    let x1 = x.narrow(1, 0, 1)?;
    let x2 = x.narrow(1, 1, 1)?;

    let ones = Tensor::ones((b_sz, 1), x.dtype(), x.device())?;
    let x1_sq = x1.mul(&x1)?;
    let x1_x2 = x1.mul(&x2)?;
    let x2_sq = x2.mul(&x2)?;
    let sin_x1 = x1.sin()?;
    let sin_x2 = x2.sin()?;

    Tensor::cat(
        &[&ones, &x1, &x2, &x1_sq, &x1_x2, &x2_sq, &sin_x1, &sin_x2],
        1,
    )
}

/// SINDy prediction and hard-thresholding step:
pub fn sindy_step(theta_x: &Tensor, xi: &Tensor, threshold: f32) -> Result<(Tensor, Tensor)> {
    let x_dot_pred = theta_x.matmul(xi)?;

    let xi_vec = xi.to_vec2::<f32>()?;
    let (rows, cols) = xi.dims2()?;
    let mut sparse_vec = vec![vec![0f32; cols]; rows];

    for r in 0..rows {
        for c in 0..cols {
            let val = xi_vec[r][c];
            if val.abs() >= threshold {
                sparse_vec[r][c] = val;
            }
        }
    }
    let mut flat = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            flat[r * cols + c] = sparse_vec[r][c];
        }
    }
    let xi_sparse = Tensor::from_vec(flat, (rows, cols), xi.device())?;
    Ok((x_dot_pred, xi_sparse))
}

/// Linearized 4D-Var Data Assimilation Analysis Step
pub fn four_d_var_step(
    x_b: &Tensor,
    b_cov_inv: &Tensor,
    y_obs: &Tensor,
    r_cov_inv: &Tensor,
    h_obs: &Tensor,
    m_dyn: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let h_t = h_obs.t()?;
    let h_m = m_dyn.matmul(&h_t)?; // [D, P]
    let h_m_t = h_m.t()?;

    // Innovation: d = y_obs - x_b @ h_m
    let pred_y = x_b.matmul(&h_m)?;
    let d_innov = y_obs.sub(&pred_y)?;

    // Hessian: B^{-1} + H_M @ R^{-1} @ H_M^T
    let hm_r = h_m.matmul(r_cov_inv)?;
    let obs_hessian = hm_r.matmul(&h_m_t)?;
    let hessian = b_cov_inv.add(&obs_hessian)?;
    let hessian_inv = invert_matrix(&hessian)?;

    // RHS: d_innov @ R^{-1} @ H_M^T
    let innov_r = d_innov.matmul(r_cov_inv)?;
    let rhs = innov_r.matmul(&h_m_t)?;

    // delta_x = rhs @ hessian_inv (since Hessian is symmetric)
    let delta_x = rhs.matmul(&hessian_inv)?;
    let x_analysis = x_b.add(&delta_x)?;

    Ok((x_analysis, delta_x))
}
