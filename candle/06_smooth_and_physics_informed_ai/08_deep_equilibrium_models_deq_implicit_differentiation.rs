//! Deep Equilibrium Models DEQ Implicit Differentiation
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Nonlinear equilibrium step: f_theta(z, x) = tanh(z @ w_z + x @ w_x + b).
pub fn f_theta(z: &Tensor, x: &Tensor, w_z: &Tensor, w_x: &Tensor, b: &Tensor) -> Result<Tensor> {
    let lin_z = z.matmul(w_z)?;
    let lin_x = x.matmul(w_x)?;
    let sum = lin_z.add(&lin_x)?.broadcast_add(b)?;
    sum.tanh()
}

/// Fixed-point Picard iteration: z_{k+1} = f_theta(z_k, x)
pub fn fixed_point_iteration(
    x: &Tensor,
    w_z: &Tensor,
    w_x: &Tensor,
    b: &Tensor,
    num_iters: usize,
) -> Result<(Tensor, Tensor)> {
    let (b_sz, _) = x.dims2()?;
    let (d_sz, _) = w_z.dims2()?;
    let mut z = Tensor::zeros((b_sz, d_sz), x.dtype(), x.device())?;

    for _ in 0..num_iters {
        z = f_theta(&z, x, w_z, w_x, b)?;
    }
    let f_val = f_theta(&z, x, w_z, w_x, b)?;
    let res = z.sub(&f_val)?;
    Ok((z, res))
}

/// Anderson acceleration (Type II) with history depth m
pub fn anderson_acceleration(
    x: &Tensor,
    w_z: &Tensor,
    w_x: &Tensor,
    b: &Tensor,
    m: usize,
    num_iters: usize,
    beta: f64,
) -> Result<(Tensor, Tensor)> {
    let (b_sz, _) = x.dims2()?;
    let (d_sz, _) = w_z.dims2()?;
    let mut z = Tensor::zeros((b_sz, d_sz), x.dtype(), x.device())?;

    let mut z_hist: Vec<Tensor> = Vec::new();
    let mut g_hist: Vec<Tensor> = Vec::new();

    for _ in 0..num_iters {
        let g = f_theta(&z, x, w_z, w_x, b)?;
        z_hist.push(z.clone());
        g_hist.push(g.clone());
        if z_hist.len() > m + 1 {
            z_hist.remove(0);
            g_hist.remove(0);
        }

        let k_hist = z_hist.len() - 1;
        if k_hist == 0 {
            z = g;
        } else {
            // Solve 2x2 or k_hist x k_hist system per batch item
            let mut z_next_rows = Vec::with_capacity(b_sz);
            for b_idx in 0..b_sz {
                // Residuals for iterates 0..=k_hist
                // R_mat of shape [D, k_hist + 1]
                let mut r_cols = Vec::new();
                for j in 0..=k_hist {
                    let g_row = g_hist[j].get(b_idx)?;
                    let z_row = z_hist[j].get(b_idx)?;
                    r_cols.push(g_row.sub(&z_row)?);
                }

                // d_R cols: R_{j} - R_{j-1} for j in 1..=k_hist
                let mut dr_cols = Vec::new();
                for j in 1..=k_hist {
                    dr_cols.push(r_cols[j].sub(&r_cols[j - 1])?);
                }

                // Build d_R matrix of shape [D, k_hist]
                // and compute lhs = d_R^T d_R + reg * I
                // For k_hist = 2:
                let mut lhs_vec = vec![0f32; k_hist * k_hist];
                let mut rhs_vec = vec![0f32; k_hist];
                let r_last = &r_cols[k_hist];

                for r in 0..k_hist {
                    // rhs[r] = dr_cols[r] . r_last
                    let dr_dot_rlast: f32 = dr_cols[r].mul(r_last)?.sum_all()?.to_scalar()?;
                    rhs_vec[r] = dr_dot_rlast;

                    for c in 0..k_hist {
                        let dot: f32 = dr_cols[r].mul(&dr_cols[c])?.sum_all()?.to_scalar()?;
                        let reg = if r == c { 1e-5f32 } else { 0f32 };
                        lhs_vec[r * k_hist + c] = dot + reg;
                    }
                }

                // Solve 2x2 or 1x1 system analytically:
                let alpha_vec = if k_hist == 1 {
                    vec![rhs_vec[0] / lhs_vec[0]]
                } else if k_hist == 2 {
                    let a = lhs_vec[0];
                    let b_elem = lhs_vec[1];
                    let c_elem = lhs_vec[2];
                    let d = lhs_vec[3];
                    let det = a * d - b_elem * c_elem;
                    let a0 = (d * rhs_vec[0] - b_elem * rhs_vec[1]) / det;
                    let a1 = (-c_elem * rhs_vec[0] + a * rhs_vec[1]) / det;
                    vec![a0, a1]
                } else {
                    panic!("Unsupported k_hist > 2 in Candle test");
                };

                // g_diff: g_{j+1} - g_j
                // z_b = g_hist[-1] - sum_j alpha_j * (g_{j+1} - g_j)
                let g_last = g_hist[k_hist].get(b_idx)?;
                let mut g_corr = Tensor::zeros((d_sz,), x.dtype(), x.device())?;
                for j in 0..k_hist {
                    let diff = g_hist[j + 1].get(b_idx)?.sub(&g_hist[j].get(b_idx)?)?;
                    let term = diff.affine(alpha_vec[j] as f64, 0.0)?;
                    g_corr = g_corr.add(&term)?;
                }
                let z_b = g_last.sub(&g_corr)?;

                // z_prev_diff: z_{j+1} - z_j
                let z_last = z_hist[k_hist].get(b_idx)?;
                let mut z_corr = Tensor::zeros((d_sz,), x.dtype(), x.device())?;
                for j in 0..k_hist {
                    let diff = z_hist[j + 1].get(b_idx)?.sub(&z_hist[j].get(b_idx)?)?;
                    let term = diff.affine(alpha_vec[j] as f64, 0.0)?;
                    z_corr = z_corr.add(&term)?;
                }
                let z_prev = z_last.sub(&z_corr)?;

                // z_elem = (1 - beta) * z_prev + beta * z_b
                let term1 = z_prev.affine(1.0 - beta, 0.0)?;
                let term2 = z_b.affine(beta, 0.0)?;
                let z_elem = term1.add(&term2)?;
                z_next_rows.push(z_elem);
            }
            z = Tensor::stack(&z_next_rows, 0)?;
        }
    }
    let f_val = f_theta(&z, x, w_z, w_x, b)?;
    let final_res = z.sub(&f_val)?;
    Ok((z, final_res))
}

/// Implicit Adjoint Vector-Jacobian Product Solve via Neumann series / fixed point
pub fn implicit_jacobian_transpose_solve(
    z_star: &Tensor,
    x: &Tensor,
    w_z: &Tensor,
    w_x: &Tensor,
    b: &Tensor,
    grad_z: &Tensor,
    num_steps: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let lin_z = z_star.matmul(w_z)?;
    let lin_x = x.matmul(w_x)?;
    let u = lin_z.add(&lin_x)?.broadcast_add(b)?;
    let tanh_u = u.tanh()?;
    let tanh_sq = tanh_u.mul(&tanh_u)?;
    let ones = Tensor::ones_like(&tanh_sq)?;
    let d_act = ones.sub(&tanh_sq)?; // 1 - tanh^2(u)

    let wz_t = w_z.t()?;
    let mut v = Tensor::zeros_like(grad_z)?;

    for _ in 0..num_steps {
        // (v * d_act) @ w_z.T
        let v_dact = v.mul(&d_act)?;
        let step = v_dact.matmul(&wz_t)?;
        v = grad_z.add(&step)?;
    }

    let delta = v.mul(&d_act)?;
    let wx_t = w_x.t()?;
    let grad_x = delta.matmul(&wx_t)?;
    let z_t = z_star.t()?;
    let grad_wz = z_t.matmul(&delta)?;

    Ok((v, grad_x, grad_wz))
}

/// Monotone Operator Splitting forward step
pub fn mon_operator_splitting(
    x: &Tensor,
    w_sym: &Tensor,
    w_skew: &Tensor,
    w_mon_x: &Tensor,
    b_mon: &Tensor,
    alpha_step: f64,
    num_iters: usize,
) -> Result<Tensor> {
    let w = w_sym.add(w_skew)?;
    let (b_sz, _) = x.dims2()?;
    let (d_sz, _) = w.dims2()?;
    let mut z = Tensor::zeros((b_sz, d_sz), x.dtype(), x.device())?;

    for _ in 0..num_iters {
        let lin_z = z.matmul(&w)?;
        let lin_x = x.matmul(w_mon_x)?;
        let lin = lin_z.add(&lin_x)?.broadcast_add(b_mon)?;
        let step = lin.affine(alpha_step, 0.0)?;
        let z_sub = z.sub(&step)?;
        z = z_sub.tanh()?;
    }
    Ok(z)
}
