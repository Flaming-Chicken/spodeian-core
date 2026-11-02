//! Native Candle implementation for Category 04 Topic 04:
//! 04_orthogonal_and_preconditioned_matrix_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use candle_core::{Result, Tensor};

/// Linear anchor module for Category 04 Topic 04
#[derive(Debug, Clone)]
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_w = x.matmul(&self.weight)?;
        x_w.broadcast_add(&self.bias)
    }
}

/// Newton-Schulz quintic iteration (Jordan et al. 2024) for matrix orthogonalization.
pub fn newton_schulz_quintic(g: &Tensor, steps: usize, eps: f64) -> Result<Tensor> {
    let (m, n) = g.dims2()?;
    let g_norm = (g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt();
    let denom = g_norm + eps;

    let mut x = (g * (1.0 / denom))?;
    let transpose = m >= n;
    if transpose {
        x = x.t()?;
    }

    let a_coeff = 3.4445f64;
    let b_coeff = -4.7750f64;
    let c_coeff = 2.0315f64;

    for _ in 0..steps {
        let a = x.matmul(&x.t()?)?;
        let a_sq = a.matmul(&a)?;
        let b_part = ((&a * b_coeff)? + (&a_sq * c_coeff)?)?;
        let update = b_part.matmul(&x)?;
        let a_x = (&x * a_coeff)?;
        x = (&a_x + &update)?;
    }

    if transpose {
        x = x.t()?;
    }

    Ok(x)
}

/// Muon optimizer step: Momentum Orthogonalized by Newton-Schulz
pub fn step_muon(
    param: &Tensor,
    grad: &Tensor,
    m_buf: &Tensor,
    lr: f64,
    momentum: f64,
    nesterov: bool,
    weight_decay: f64,
    ns_steps: usize,
) -> Result<(Tensor, Tensor)> {
    let mom_m = (m_buf * momentum)?;
    let new_m = (&mom_m + grad)?;

    let update_cand = if nesterov {
        let n_mom = (&new_m * momentum)?;
        (&n_mom + grad)?
    } else {
        new_m.clone()
    };

    let ortho_update = newton_schulz_quintic(&update_cand, ns_steps, 1e-7)?;
    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let scaled_update = (&ortho_update * lr)?;
    let new_p = (&shrunk_p - &scaled_update)?;

    Ok((new_p, new_m))
}

/// Computes M^(-1/4) for a symmetric positive definite matrix using Jacobi diagonalization.
pub fn matrix_power_neg_fourth(m: &Tensor, eps: f64) -> Result<Tensor> {
    let (dim_r, dim_c) = m.dims2()?;
    assert_eq!(dim_r, dim_c);
    let n = dim_r;

    let flat_m = m.to_vec2::<f32>()?;
    let mut a = vec![vec![0.0f64; n]; n];
    let mut v = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        v[i][i] = 1.0;
        for j in 0..n {
            a[i][j] = flat_m[i][j] as f64;
        }
    }

    // Jacobi eigenvalue algorithm
    for _ in 0..60 {
        let mut p = 0;
        let mut q = 1;
        let mut max_val = 0.0f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply Givens rotation to A: J^T A J
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = c * a[i][p] - s * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = s * a[i][p] + c * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        new_a[q][q] = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Accumulate eigenvectors in V: V * J
        for i in 0..n {
            let v_ip = v[i][p];
            let v_iq = v[i][q];
            v[i][p] = c * v_ip - s * v_iq;
            v[i][q] = s * v_ip + c * v_iq;
        }
    }

    // Reconstruct V @ diag(lambda^(-1/4)) @ V^T
    let mut inv_d = vec![0.0f64; n];
    for i in 0..n {
        let val = a[i][i].max(eps);
        inv_d[i] = val.powf(-0.25);
    }

    let mut res = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += v[i][k] * inv_d[k] * v[j][k];
            }
            res[i][j] = sum as f32;
        }
    }

    Tensor::new(res, m.device())
}

/// Shampoo optimizer step (Gupta et al. 2018)
pub fn step_shampoo(
    param: &Tensor,
    grad: &Tensor,
    l_cov: &Tensor,
    r_cov: &Tensor,
    lr: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let grad_gt = grad.matmul(&grad.t()?)?;
    let grad_tg = grad.t()?.matmul(grad)?;

    let new_l = ((l_cov * beta2)? + (&grad_gt * (1.0 - beta2))?)?;
    let new_r = ((r_cov * beta2)? + (&grad_tg * (1.0 - beta2))?)?;

    let l_inv = matrix_power_neg_fourth(&new_l, eps)?;
    let r_inv = matrix_power_neg_fourth(&new_r, eps)?;

    let precond_grad = l_inv.matmul(grad)?.matmul(&r_inv)?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let step_p = (&precond_grad * lr)?;
    let new_p = (&shrunk_p - &step_p)?;

    Ok((new_p, new_l, new_r))
}

/// GaLore optimizer step (Zhao et al. 2024)
pub fn step_galore(
    param: &Tensor,
    grad: &Tensor,
    m_low: &Tensor,
    v_low: &Tensor,
    proj_matrix: &Tensor,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let grad_low = proj_matrix.t()?.matmul(grad)?;

    let new_m = ((m_low * beta1)? + (&grad_low * (1.0 - beta1))?)?;
    let new_v = ((v_low * beta2)? + (grad_low.sqr()? * (1.0 - beta2))?)?;

    let bc1 = 1.0 - (beta1 as f64).powi(step as i32);
    let bc2 = 1.0 - (beta2 as f64).powi(step as i32);

    let m_hat = (&new_m * (1.0 / bc1))?;
    let v_hat = (&new_v * (1.0 / bc2))?;

    let denom = v_hat.sqrt()?.affine(1.0, eps)?;
    let adam_step = m_hat.div(&denom)?;

    let full_update = proj_matrix.matmul(&adam_step)?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let step_p = (&full_update * lr)?;
    let new_p = (&shrunk_p - &step_p)?;

    Ok((new_p, new_m, new_v))
}
