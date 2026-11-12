//! Topic 11: Randomized Numerical Linear Algebra, Spectral Autograd & Hutch++.
//!
//! Provides native Candle implementations for:
//! 1. Modified Gram-Schmidt thin QR decomposition with strictly positive diagonal R.
//! 2. Halko-Martinsson-Tropp randomized range finder.
//! 3. Meyer-Musco-Musco-Woodruff Hutch++ stochastic trace estimator combining low-rank sketching with Monte Carlo.
//! 4. Lorentzian-broadened spectral autograd coupling matrix avoiding degenerate singular value NaN blowups.
//! 5. Power iteration for spectral norm computation.
//! 6. `SpectralLinear` projection module.
//!
//! # Mathematical Invariants
//! - Modified Gram-Schmidt produces uniquely sign-determined orthonormal basis $Q^T Q = I_l$ with $R_{jj} > 0$.
//! - Hutch++ achieves $\mathcal{O}(1/\epsilon)$ matrix-vector queries vs. classical Hutchinson $\mathcal{O}(1/\epsilon^2)$.
//! - Lorentzian broadening $F_{ij} = \frac{\sigma_j - \sigma_i}{(\sigma_j - \sigma_i)^2 + \epsilon^2}$ is smooth and $C^\infty$ everywhere.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp::{
//!     modified_gram_schmidt_qr, lorentzian_broadened_f_matrix,
//! };
//!
//! let device = Device::Cpu;
//! let y = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).unwrap();
//! let (q, r) = modified_gram_schmidt_qr(&y, &device).unwrap();
//! assert_eq!(q.dims(), &[3, 2]);
//! assert_eq!(r.dims(), &[2, 2]);
//!
//! let sigma = Tensor::new(&[5.0f32, 3.0, 3.0, 1.0], &device).unwrap();
//! let f = lorentzian_broadened_f_matrix(&sigma, 0.01, &device).unwrap();
//! assert_eq!(f.dims(), &[4, 4]);
//! ```

use candle_core::{Device, Result, Tensor};

/// Thin QR decomposition $Y = Q R$ via Modified Gram-Schmidt with strictly positive diagonal $R_{jj}$.
pub fn modified_gram_schmidt_qr(y: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
    let dims = y.dims();
    let m = dims[0];
    let l = dims[1];
    let y_flat = y.to_vec2::<f32>()?;

    // q columns stored row-by-row for each column: q_cols[j][i] is Q[i, j]
    let mut q_cols: Vec<Vec<f32>> = (0..l)
        .map(|j| (0..m).map(|i| y_flat[i][j]).collect())
        .collect();
    let mut r_mat = vec![vec![0.0f32; l]; l];

    for j in 0..l {
        for i in 0..j {
            let mut dot = 0.0f32;
            for k in 0..m {
                dot += q_cols[i][k] * q_cols[j][k];
            }
            r_mat[i][j] = dot;
            for k in 0..m {
                q_cols[j][k] -= dot * q_cols[i][k];
            }
        }
        let mut norm_sq = 0.0f32;
        for k in 0..m {
            norm_sq += q_cols[j][k] * q_cols[j][k];
        }
        let norm = norm_sq.sqrt();
        r_mat[j][j] = norm;
        let scale = 1.0f32 / (norm + 1e-12);
        for k in 0..m {
            q_cols[j][k] *= scale;
        }
    }

    // Reconstruct Q matrix [m, l]
    let mut q_mat = vec![0.0f32; m * l];
    for i in 0..m {
        for j in 0..l {
            q_mat[i * l + j] = q_cols[j][i];
        }
    }

    // Flatten R matrix [l, l]
    let mut r_flat = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            r_flat[i * l + j] = r_mat[i][j];
        }
    }

    let q_t = Tensor::from_vec(q_mat, (m, l), device)?;
    let r_t = Tensor::from_vec(r_flat, (l, l), device)?;
    Ok((q_t, r_t))
}

/// Computes orthonormal basis $Q$ spanning the dominant range of $A$ via sketch $Y = A \Omega$.
pub fn randomized_range_finder(a: &Tensor, omega: &Tensor, device: &Device) -> Result<Tensor> {
    let y = a.matmul(omega)?;
    let (q, _) = modified_gram_schmidt_qr(&y, device)?;
    Ok(q)
}

/// Evaluates Hutch++ stochastic trace estimator combining low-rank projection with Monte Carlo.
pub fn hutch_plus_plus_trace(
    a: &Tensor,
    s_sketch: &Tensor,
    g_mc: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let dims_g = g_mc.dims();
    let n = dims_g[0];
    let m = dims_g[1] as f32;

    // 1. Low-rank subspace sketch
    let y = a.matmul(s_sketch)?;
    let (q, _) = modified_gram_schmidt_qr(&y, device)?;

    // 2. Exact trace of projection on Q: tr(Q^T A Q)
    let q_t = q.t()?;
    let q_t_a_q = q_t.matmul(a)?.matmul(&q)?;
    let q_dims = q_t_a_q.dims();
    let l = q_dims[0];
    let q_flat = q_t_a_q.to_vec2::<f32>()?;
    let mut tr_proj = 0.0f32;
    for i in 0..l {
        tr_proj += q_flat[i][i];
    }

    // 3. Residual Monte Carlo on (I - Q Q^T) A
    let a_g = a.matmul(g_mc)?;
    let proj_g = q.matmul(&q_t.matmul(&a_g)?)?;
    let residual_g = a_g.sub(&proj_g)?;

    let g_flat = g_mc.to_vec2::<f32>()?;
    let res_flat = residual_g.to_vec2::<f32>()?;
    let m_cols = dims_g[1];

    let mut tr_res = 0.0f32;
    for col in 0..m_cols {
        let mut dot = 0.0f32;
        for row in 0..n {
            dot += g_flat[row][col] * res_flat[row][col];
        }
        tr_res += dot;
    }
    tr_res /= m;

    Tensor::from_vec(vec![tr_proj + tr_res], (1,), device)
}

/// Skew-symmetric Lorentzian-broadened singular value coupling matrix $F_{ij}$.
pub fn lorentzian_broadened_f_matrix(
    singular_values: &Tensor,
    epsilon: f32,
    device: &Device,
) -> Result<Tensor> {
    let s = singular_values.to_vec1::<f32>()?;
    let n = s.len();
    let eps_sq = epsilon * epsilon;

    let mut f_data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let delta = s[j] - s[i];
                f_data[i * n + j] = delta / (delta * delta + eps_sq);
            }
        }
    }

    Tensor::from_vec(f_data, (n, n), device)
}

/// Dominant singular value and vectors via Power Iteration.
pub fn power_iteration_spectral_norm(
    w: &Tensor,
    v_init: &Tensor,
    num_iters: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims_w = w.dims();
    let m = dims_w[0];
    let _n = dims_w[1];

    let w_t = w.t()?;
    let mut v = v_init.clone();
    let norm_v = v.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

    v = (v / (norm_v as f64 + 1e-12))?;

    let mut u = Tensor::zeros((m,), candle_core::DType::F32, device)?;

    for _ in 0..num_iters {
        let u_unnorm = w.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
        let norm_u = u_unnorm.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        u = (u_unnorm / (norm_u as f64 + 1e-12))?;

        let v_unnorm = w_t.matmul(&u.unsqueeze(1)?)?.squeeze(1)?;
        let norm_v_next = v_unnorm.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        v = (v_unnorm / (norm_v_next as f64 + 1e-12))?;
    }

    let w_v = w.matmul(&v.unsqueeze(1)?)?.squeeze(1)?;
    let sigma_val = u.mul(&w_v)?.sum_all()?.to_scalar::<f32>()?;
    let sigma = Tensor::from_vec(vec![sigma_val], (1,), device)?;

    Ok((sigma, u, v))
}

/// Spectral linear projection layer.
#[derive(Debug, Clone)]
pub struct SpectralLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl SpectralLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight)?;
        y.broadcast_add(&self.bias)
    }

    pub fn compute_spectral_norm(
        &self,
        v_init: &Tensor,
        num_iters: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let (sigma, _, _) = power_iteration_spectral_norm(&self.weight, v_init, num_iters, device)?;
        Ok(sigma)
    }
}
