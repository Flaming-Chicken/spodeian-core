//! AI Analysis Loss Landscape Hessian Lanczos and Empirical NTK
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Filter-normalization for scale-invariant loss landscape visualization (Li et al., 2018).
pub fn filter_normalize_direction(theta: &Tensor, delta: &Tensor) -> Result<Tensor> {
    let num_filters = theta.dim(0)?;
    let filter_dim = theta.dim(1)?;
    let th = theta.to_vec2::<f32>()?;
    let dt = delta.to_vec2::<f32>()?;

    let mut out = vec![vec![0.0f32; filter_dim]; num_filters];
    for i in 0..num_filters {
        let mut norm_th_sq = 0.0f32;
        let mut norm_dt_sq = 0.0f32;
        for j in 0..filter_dim {
            norm_th_sq += th[i][j] * th[i][j];
            norm_dt_sq += dt[i][j] * dt[i][j];
        }
        let norm_th = (norm_th_sq + 1e-8).sqrt();
        let norm_dt = (norm_dt_sq + 1e-8).sqrt();
        let scale = norm_th / norm_dt;
        for j in 0..filter_dim {
            out[i][j] = dt[i][j] * scale;
        }
    }
    Tensor::new(out, theta.device())
}

/// Evaluates sliced 2D coordinate on loss landscape: theta(alpha, beta) = theta + alpha * d1 + beta * d2.
pub fn loss_surface_coord(
    theta: &Tensor,
    d1: &Tensor,
    d2: &Tensor,
    alpha: f32,
    beta: f32,
) -> Result<Tensor> {
    let t1 = d1.affine(alpha as f64, 0.0)?;
    let t2 = d2.affine(beta as f64, 0.0)?;
    theta.add(&t1)?.add(&t2)
}

/// Lanczos iteration for Hessian spectral estimation (Ghorbani et al., 2019 / PyHessian).
pub fn lanczos_tridiagonal(h: &Tensor, v0: &Tensor, num_steps: usize) -> Result<(Tensor, Tensor)> {
    let p = h.dim(0)?;
    let h_mat = h.to_vec2::<f32>()?;
    let v_vec = v0.to_vec1::<f32>()?;

    let mut v_norm_sq = 0.0f32;
    for &val in &v_vec {
        v_norm_sq += val * val;
    }
    let v_norm = (v_norm_sq + 1e-8).sqrt();

    let mut q = vec![vec![0.0f32; p]; num_steps + 1];
    for i in 0..p {
        q[0][i] = v_vec[i] / v_norm;
    }

    let mut alpha_diag = vec![0.0f32; num_steps];
    let mut beta_diag = vec![0.0f32; num_steps - 1];

    for j in 0..num_steps {
        // w = H * q[j]
        let mut w = vec![0.0f32; p];
        for r in 0..p {
            let mut sum = 0.0f32;
            for c in 0..p {
                sum += h_mat[r][c] * q[j][c];
            }
            w[r] = sum;
        }

        if j > 0 {
            for i in 0..p {
                w[i] -= beta_diag[j - 1] * q[j - 1][i];
            }
        }

        let mut a_j = 0.0f32;
        for i in 0..p {
            a_j += q[j][i] * w[i];
        }
        alpha_diag[j] = a_j;

        for i in 0..p {
            w[i] -= a_j * q[j][i];
        }

        if j < num_steps - 1 {
            let mut b_sq = 0.0f32;
            for i in 0..p {
                b_sq += w[i] * w[i];
            }
            let b_j = (b_sq + 1e-8).sqrt();
            beta_diag[j] = b_j;
            for i in 0..p {
                q[j + 1][i] = w[i] / b_j;
            }
        }
    }

    Ok((
        Tensor::new(alpha_diag, h.device())?,
        Tensor::new(beta_diag, h.device())?,
    ))
}

/// Empirical Neural Tangent Kernel (eNTK) Gram matrix Theta = J J^T (Jacot et al., 2018).
pub fn empirical_ntk(jacobians: &Tensor) -> Result<Tensor> {
    jacobians.matmul(&jacobians.t()?)
}
