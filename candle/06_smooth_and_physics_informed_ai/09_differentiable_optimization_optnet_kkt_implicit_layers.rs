//! Differentiable Optimization OptNet KKT Implicit Layers
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Invert an N x N matrix (for N <= 4) analytically or via Gauss-Jordan elimination on CPU.
fn invert_matrix(mat: &Tensor) -> Result<Tensor> {
    let (n, n2) = mat.dims2()?;
    assert_eq!(n, n2);
    let flat = mat.to_vec2::<f32>()?;

    // Augmented matrix [A | I]
    let mut aug = vec![vec![0f32; 2 * n]; n];
    for r in 0..n {
        for c in 0..n {
            aug[r][c] = flat[r][c];
        }
        aug[r][n + r] = 1.0f32;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Pivot
        let mut pivot = aug[i][i];
        if pivot.abs() < 1e-9 {
            // Find non-zero row
            for r in (i + 1)..n {
                if aug[r][i].abs() > pivot.abs() {
                    aug.swap(i, r);
                    pivot = aug[i][i];
                    break;
                }
            }
        }
        assert!(pivot.abs() > 1e-9, "Singular matrix in inversion");
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

    // Extract inverse
    let mut inv_flat = vec![0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            inv_flat[r * n + c] = aug[r][n + c];
        }
    }
    Tensor::from_vec(inv_flat, (n, n), mat.device())
}

/// Solve strictly convex equality-constrained QP per batch:
/// min_z (1/2) z^T Q z + p^T z  s.t.  A z = b.
pub fn solve_equality_qp(
    q: &Tensor,
    p: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (_b_sz, d) = p.dims2()?;
    let (m, d2) = a.dims2()?;
    assert_eq!(d, d2);
    let n = d + m;

    // Build KKT matrix
    // [ Q   A^T ]
    // [ A    0  ]
    let q_mat = q.to_vec2::<f32>()?;
    let a_mat = a.to_vec2::<f32>()?;
    let mut kkt_vec = vec![0f32; n * n];

    for r in 0..d {
        for c in 0..d {
            kkt_vec[r * n + c] = q_mat[r][c];
        }
        for c in 0..m {
            kkt_vec[r * n + (d + c)] = a_mat[c][r]; // A^T
        }
    }
    for r in 0..m {
        for c in 0..d {
            kkt_vec[(d + r) * n + c] = a_mat[r][c]; // A
        }
    }

    let kkt = Tensor::from_vec(kkt_vec, (n, n), q.device())?;
    let kkt_inv = invert_matrix(&kkt)?;

    // RHS: [-p, b] shape [B, n]
    let neg_p = p.affine(-1.0, 0.0)?;
    let rhs = Tensor::cat(&[&neg_p, b], 1)?;

    // Sol = RHS @ KKT_inv^T (or KKT_inv since KKT is symmetric)
    let sol = rhs.matmul(&kkt_inv)?;
    let z_star = sol.narrow(1, 0, d)?;
    let nu_star = sol.narrow(1, d, m)?;

    Ok((z_star, nu_star))
}

/// Backward step through KKT conditions
pub fn optnet_backward_step(
    q: &Tensor,
    a: &Tensor,
    grad_z: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b_sz, d) = grad_z.dims2()?;
    let (m, _) = a.dims2()?;
    let n = d + m;

    let q_mat = q.to_vec2::<f32>()?;
    let a_mat = a.to_vec2::<f32>()?;
    let mut kkt_vec = vec![0f32; n * n];

    for r in 0..d {
        for c in 0..d {
            kkt_vec[r * n + c] = q_mat[r][c];
        }
        for c in 0..m {
            kkt_vec[r * n + (d + c)] = a_mat[c][r];
        }
    }
    for r in 0..m {
        for c in 0..d {
            kkt_vec[(d + r) * n + c] = a_mat[r][c];
        }
    }

    let kkt = Tensor::from_vec(kkt_vec, (n, n), q.device())?;
    let kkt_inv = invert_matrix(&kkt)?;

    let neg_grad_z = grad_z.affine(-1.0, 0.0)?;
    let zeros_m = Tensor::zeros((b_sz, m), grad_z.dtype(), grad_z.device())?;
    let rhs_adj = Tensor::cat(&[&neg_grad_z, &zeros_m], 1)?;

    let sol_adj = rhs_adj.matmul(&kkt_inv)?;
    let d_z = sol_adj.narrow(1, 0, d)?;
    let d_nu = sol_adj.narrow(1, d, m)?;

    let grad_p = d_z.affine(-1.0, 0.0)?;
    let grad_b = d_nu.affine(-1.0, 0.0)?;

    Ok((d_z, grad_p, grad_b))
}

/// Log-barrier smoothed interior point QP solve
pub fn barrier_qp_solve(
    q: &Tensor,
    p: &Tensor,
    g: &Tensor,
    h_bound: &Tensor,
    num_steps: usize,
    eta: f64,
    t_barrier: f64,
) -> Result<(Tensor, Tensor)> {
    let (b_sz, d) = p.dims2()?;
    let mut z = Tensor::zeros((b_sz, d), p.dtype(), p.device())?;
    let g_t = g.t()?;

    for _ in 0..num_steps {
        let z_gt = z.matmul(&g_t)?;
        let slacks = h_bound.sub(&z_gt)?;
        let safe_slacks = slacks.clamp(1e-3f32, 1e9f32)?;
        let inv_slacks = safe_slacks.recip()?;
        let dual_lambda = inv_slacks.affine(1.0 / t_barrier, 0.0)?;
        let grad_barrier = dual_lambda.matmul(g)?;
        let grad_q = z.matmul(q)?;
        let grad_total = grad_q.add(p)?.add(&grad_barrier)?;
        let step = grad_total.affine(eta, 0.0)?;
        z = z.sub(&step)?;
    }

    let z_gt = z.matmul(&g_t)?;
    let final_slacks = h_bound.sub(&z_gt)?.clamp(1e-4f32, 1e9f32)?;
    let final_lambda = final_slacks.recip()?.affine(1.0 / t_barrier, 0.0)?;

    Ok((z, final_lambda))
}
