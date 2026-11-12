//! Physics Informed PINNs Deep Ritz Hard Bc and NTK Balancing
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Evaluates network forward pass and analytical first and second derivatives.
///
/// Returns (u_raw, u_x, u_xx, phi).
pub fn pinn_forward_and_derivatives(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let z = x.matmul(w1)?.broadcast_add(b1)?;
    let a = z.tanh()?;
    let u_raw = a.matmul(w2)?.broadcast_add(b2)?;

    // 1 - a^2
    let a_sqr = a.sqr()?;
    let one = Tensor::ones_like(&a)?;
    let one_minus_a2 = (one - a_sqr)?;

    // psi = (1 - a^2) * w1 -> u_x = psi @ w2
    let psi = one_minus_a2.broadcast_mul(w1)?;
    let u_x = psi.matmul(w2)?;

    // phi = -2 * a * (1 - a^2) * (w1^2) -> u_xx = phi @ w2
    let w1_sqr = w1.sqr()?;
    let neg_two_a = a.affine(-2.0, 0.0)?;
    let phi = neg_two_a.mul(&one_minus_a2)?.broadcast_mul(&w1_sqr)?;
    let u_xx = phi.matmul(w2)?;

    Ok((u_raw, u_x, u_xx, phi))
}

/// Computes PDE residual r(x) = -u_xx - pi^2 * sin(pi * x) and MSE loss.
pub fn pde_residual_and_loss(x: &Tensor, u_xx: &Tensor) -> Result<(Tensor, Tensor)> {
    let pi = std::f64::consts::PI;
    let pi_x = x.affine(pi, 0.0)?;
    let sin_pi_x = pi_x.sin()?;
    let f_source = sin_pi_x.affine(pi * pi, 0.0)?;

    let neg_u_xx = u_xx.neg()?;
    let pde_residual = (neg_u_xx - f_source)?;
    let loss_pde = pde_residual.sqr()?.mean_all()?.reshape(&[1])?;

    Ok((pde_residual, loss_pde))
}

/// Implements Hard Boundary Conditions: u_hard(x) = d(x) * u(x) with d(x) = x * (1 - x).
pub fn hard_boundary_conditions(
    x: &Tensor,
    u_raw: &Tensor,
    u_x: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let one = Tensor::ones_like(x)?;
    let one_minus_x = (&one - x)?;
    let d = x.mul(&one_minus_x)?;

    // d'(x) = 1 - 2x
    let two_x = x.affine(2.0, 0.0)?;
    let d_x = (&one - two_x)?;

    let u_hard = d.mul(u_raw)?;
    let term1 = d_x.mul(u_raw)?;
    let term2 = d.mul(u_x)?;
    let u_hard_x = (term1 + term2)?;

    Ok((u_hard, u_hard_x))
}

/// Evaluates Variational / Weak-Form Deep Ritz Energy:
/// E(u) = (1/N) * sum(0.5 * (u_hard_x)^2 - f * u_hard)
pub fn deep_ritz_energy(x: &Tensor, u_hard: &Tensor, u_hard_x: &Tensor) -> Result<Tensor> {
    let pi = std::f64::consts::PI;
    let pi_x = x.affine(pi, 0.0)?;
    let f_source = pi_x.sin()?.affine(pi * pi, 0.0)?;

    let half_grad_sq = u_hard_x.sqr()?.affine(0.5, 0.0)?;
    let f_u = f_source.mul(u_hard)?;
    let energy_density = (half_grad_sq - f_u)?;
    let ritz_energy = energy_density.mean_all()?.reshape(&[1])?;

    Ok(ritz_energy)
}

/// Computes soft boundary loss and NTK eigenvalue gradient balancing between PDE and BC objectives.
pub fn ntk_loss_and_balancing(
    x_colloc: &Tensor,
    phi: &Tensor,
    pde_residual: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let n = x_colloc.dim(0)? as f64;
    // g_pde = (-2 / N) * phi^T @ pde_residual
    let g_pde = phi.t()?.matmul(pde_residual)?.affine(-2.0 / n, 0.0)?;

    // Boundary points x_bc = [0.0, 1.0]
    let x_bc = Tensor::new(&[[0.0f32], [1.0f32]], x_colloc.device())?;
    let z_bc = x_bc.matmul(w1)?.broadcast_add(b1)?;
    let a_bc = z_bc.tanh()?;
    let u_bc = a_bc.matmul(w2)?.broadcast_add(b2)?;
    let loss_bc = u_bc.sqr()?.mean_all()?.reshape(&[1])?;

    // g_bc = (2 / N_bc) * a_bc^T @ u_bc
    let n_bc = 2.0;
    let g_bc = a_bc.t()?.matmul(&u_bc)?.affine(2.0 / n_bc, 0.0)?;

    let norm_pde = (g_pde.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt();
    let norm_bc = (g_bc.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt();
    let lam = norm_pde / (norm_bc + 1e-8);
    let lambda_ntk = Tensor::new(&[lam as f32], x_colloc.device())?;

    let g_balanced = (&g_pde + g_bc.affine(lam, 0.0)?)?;

    Ok((loss_bc, g_pde, g_bc, lambda_ntk, g_balanced))
}
