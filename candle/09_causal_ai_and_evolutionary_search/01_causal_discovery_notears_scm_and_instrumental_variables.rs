//! Causal Discovery NOTEARS SCM and Instrumental Variables
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Zheng et al. (2018) NOTEARS continuous characterization of acyclicity:
/// h(W) = Tr( exp(W \odot W) ) - d
/// grad_h(W) = 2.0 * (exp(W \odot W))^T \odot W
pub fn notears_acyclicity_constraint(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (d, _) = w.dims2()?;
    let w_data = w.to_vec2::<f32>()?;

    let mut m = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            m[i][j] = w_data[i][j] * w_data[i][j];
        }
    }

    // Matrix exponential via Taylor expansion
    let mut exp_m = vec![vec![0.0f32; d]; d];
    let mut term = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        exp_m[i][i] = 1.0;
        term[i][i] = 1.0;
    }

    for k in 1..=14 {
        let mut next_term = vec![vec![0.0f32; d]; d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0f32;
                for p in 0..d {
                    s += term[i][p] * m[p][j];
                }
                next_term[i][j] = s / (k as f32);
            }
        }
        term = next_term;
        for i in 0..d {
            for j in 0..d {
                exp_m[i][j] += term[i][j];
            }
        }
    }

    let mut trace = 0.0f32;
    for i in 0..d {
        trace += exp_m[i][i];
    }
    let h = trace - (d as f32);

    let mut grad_h = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            grad_h[i][j] = 2.0 * exp_m[j][i] * w_data[i][j];
        }
    }

    let h_t = Tensor::from_vec(vec![h], 1, w.device())?;
    let grad_flat: Vec<f32> = grad_h.into_iter().flatten().collect();
    let grad_t = Tensor::from_vec(grad_flat, (d, d), w.device())?;

    Ok((h_t, grad_t))
}

fn mat_mul_4x4(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0f32;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// SCM forward generation and counterfactual intervention.
pub fn scm_generate_and_counterfactual(
    scm_w: &Tensor,
    scm_u: &Tensor,
    do_var_idx: usize,
    do_val: f32,
) -> Result<(Tensor, Tensor)> {
    let w_vec = scm_w.to_vec2::<f32>()?;
    let u_vec = scm_u.to_vec2::<f32>()?;

    let mut w = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            w[i][j] = w_vec[i][j];
        }
    }

    // For 4x4 DAG, (I - W)^-1 = I + W + W^2 + W^3
    let mut eye = [[0.0f32; 4]; 4];
    for i in 0..4 {
        eye[i][i] = 1.0;
    }
    let w2 = mat_mul_4x4(&w, &w);
    let w3 = mat_mul_4x4(&w2, &w);

    let mut inv_w = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv_w[i][j] = eye[i][j] + w[i][j] + w2[i][j] + w3[i][j];
        }
    }

    // Factual: x = u @ inv_w
    let mut x_factual = [0.0f32; 4];
    for j in 0..4 {
        for i in 0..4 {
            x_factual[j] += u_vec[0][i] * inv_w[i][j];
        }
    }

    // Intervene: sever incoming edges to do_var_idx and set u[do_var_idx] = do_val
    let mut w_int = w;
    for i in 0..4 {
        w_int[i][do_var_idx] = 0.0;
    }
    let w_int2 = mat_mul_4x4(&w_int, &w_int);
    let w_int3 = mat_mul_4x4(&w_int2, &w_int);

    let mut inv_w_int = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            inv_w_int[i][j] = eye[i][j] + w_int[i][j] + w_int2[i][j] + w_int3[i][j];
        }
    }

    let mut u_int = u_vec[0].clone();
    u_int[do_var_idx] = do_val;

    let mut x_cf = [0.0f32; 4];
    for j in 0..4 {
        for i in 0..4 {
            x_cf[j] += u_int[i] * inv_w_int[i][j];
        }
    }

    let fact_t = Tensor::from_vec(x_factual.to_vec(), (1, 4), scm_w.device())?;
    let cf_t = Tensor::from_vec(x_cf.to_vec(), (1, 4), scm_w.device())?;
    Ok((fact_t, cf_t))
}

/// Two-Stage Least Squares (2SLS) Instrumental Variables estimator.
pub fn two_stage_least_squares(
    z_inst: &Tensor,
    t_treat: &Tensor,
    y_outcome: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (n, k) = z_inst.dims2()?;
    let z = z_inst.to_vec2::<f32>()?;
    let t = t_treat.to_vec2::<f32>()?;
    let y = y_outcome.to_vec2::<f32>()?;

    // Z^T Z (2x2)
    let mut zt_z = vec![vec![0.0f32; k]; k];
    for r in 0..k {
        for c in 0..k {
            let mut s = 0.0f32;
            for i in 0..n {
                s += z[i][r] * z[i][c];
            }
            zt_z[r][c] = s;
        }
    }

    // Invert 2x2 matrix
    let det = zt_z[0][0] * zt_z[1][1] - zt_z[0][1] * zt_z[1][0];
    let inv_det = 1.0f32 / det;
    let inv_zt_z = [
        [zt_z[1][1] * inv_det, -zt_z[0][1] * inv_det],
        [-zt_z[1][0] * inv_det, zt_z[0][0] * inv_det],
    ];

    // Z^T T (2x1)
    let mut zt_t = [0.0f32; 2];
    for r in 0..2 {
        for i in 0..n {
            zt_t[r] += z[i][r] * t[i][0];
        }
    }

    // gamma = inv(Z^T Z) @ (Z^T T)
    let gamma = [
        inv_zt_z[0][0] * zt_t[0] + inv_zt_z[0][1] * zt_t[1],
        inv_zt_z[1][0] * zt_t[0] + inv_zt_z[1][1] * zt_t[1],
    ];

    // T_hat = Z @ gamma (nx1)
    let mut t_hat = Vec::with_capacity(n);
    for i in 0..n {
        t_hat.push(z[i][0] * gamma[0] + z[i][1] * gamma[1]);
    }

    // beta = (T_hat^T Y) / (T_hat^T T_hat)
    let mut th_y = 0.0f32;
    let mut th_th = 0.0f32;
    for i in 0..n {
        th_y += t_hat[i] * y[i][0];
        th_th += t_hat[i] * t_hat[i];
    }
    let beta = th_y / th_th;

    let that_t = Tensor::from_vec(t_hat, (n, 1), z_inst.device())?;
    let beta_t = Tensor::from_vec(vec![beta], (1, 1), z_inst.device())?;
    Ok((that_t, beta_t))
}
