//! MCMC, Hamiltonian Monte Carlo, SGLD, and Stein Variational Gradient Descent in Candle.
//!
//! Features:
//! - Hamiltonian Monte Carlo (HMC) with symplectic leapfrog integrator
//! - Stochastic Gradient Langevin Dynamics (SGLD) with thermal noise injection
//! - Stein Variational Gradient Descent (SVGD) RBF velocity field

use candle_core::{Result, Tensor};

/// Runs L symplectic leapfrog integration steps on quadratic potential:
/// U(theta) = 0.5 * theta^T Sigma_inv theta
pub fn hmc_leapfrog(
    theta: &Tensor,
    p: &Tensor,
    sigma_inv: &Tensor,
    eps: f64,
    l_steps: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut cur_theta = theta.clone();
    let mut cur_p = p.clone();

    for _ in 0..l_steps {
        // Half step momentum
        let theta_col = cur_theta.unsqueeze(1)?;
        let grad_u = sigma_inv.matmul(&theta_col)?.squeeze(1)?;
        cur_p = (&cur_p - grad_u.affine(0.5 * eps, 0.0)?)?;

        // Full step position
        cur_theta = (&cur_theta + cur_p.affine(eps, 0.0)?)?;

        // Half step momentum
        let theta_col_new = cur_theta.unsqueeze(1)?;
        let grad_u_new = sigma_inv.matmul(&theta_col_new)?.squeeze(1)?;
        cur_p = (&cur_p - grad_u_new.affine(0.5 * eps, 0.0)?)?;
    }

    // Hamiltonian H = 0.5 * theta^T Sigma_inv theta + 0.5 * p^T p
    let theta_col_init = theta.unsqueeze(1)?;
    let u_init = (theta
        .unsqueeze(0)?
        .matmul(&sigma_inv.matmul(&theta_col_init)?)?
        .squeeze(0)?
        .squeeze(0)?
        .to_scalar::<f32>()?
        * 0.5) as f64;
    let k_init = (p.sqr()?.sum_all()?.to_scalar::<f32>()? * 0.5) as f64;
    let h_init = u_init + k_init;

    let theta_col_prop = cur_theta.unsqueeze(1)?;
    let u_prop = (cur_theta
        .unsqueeze(0)?
        .matmul(&sigma_inv.matmul(&theta_col_prop)?)?
        .squeeze(0)?
        .squeeze(0)?
        .to_scalar::<f32>()?
        * 0.5) as f64;
    let k_prop = (cur_p.sqr()?.sum_all()?.to_scalar::<f32>()? * 0.5) as f64;
    let h_prop = u_prop + k_prop;

    let accept = (h_init - h_prop).exp().min(1.0);
    let accept_t = Tensor::new(&[accept as f32], theta.device())?;

    Ok((cur_theta, cur_p, accept_t))
}

/// Executes single SGLD step:
/// theta_{t+1} = theta_t + 0.5 * eps * grad_logp + sqrt(eps) * noise
pub fn sgld_step(theta: &Tensor, grad_logp: &Tensor, noise: &Tensor, eps: f64) -> Result<Tensor> {
    let drift = grad_logp.affine(0.5 * eps, 0.0)?;
    let thermal = noise.affine(eps.sqrt(), 0.0)?;
    (theta + drift)? + thermal
}

/// Computes SVGD velocity field:
/// phi(x) = (1/M) sum_j [k(x_j, x) grad log p(x_j) + grad_{x_j} k(x_j, x)]
pub fn svgd_velocity_field(
    particles: &Tensor,
    grad_logp: &Tensor,
    h: f64,
) -> Result<(Tensor, Tensor)> {
    let (m, d) = particles.dims2()?;
    let part_vec: Vec<Vec<f32>> = particles.to_vec2()?;
    let grad_vec: Vec<Vec<f32>> = grad_logp.to_vec2()?;

    let mut k_mat = vec![vec![0.0f32; m]; m];
    let h_sq = (h * h) as f32;

    for i in 0..m {
        for j in 0..m {
            let mut dist_sq = 0.0f32;
            for c in 0..d {
                let diff = part_vec[i][c] - part_vec[j][c];
                dist_sq += diff * diff;
            }
            k_mat[i][j] = (-0.5f32 * dist_sq / h_sq).exp();
        }
    }

    let mut phi = vec![vec![0.0f32; d]; m];
    let m_f32 = m as f32;

    for i in 0..m {
        for c in 0..d {
            let mut drive_c = 0.0f32;
            let mut rep_c = 0.0f32;
            for j in 0..m {
                drive_c += k_mat[i][j] * grad_vec[j][c];
                let diff = part_vec[i][c] - part_vec[j][c];
                rep_c += (diff / h_sq) * k_mat[i][j];
            }
            phi[i][c] = (drive_c + rep_c) / m_f32;
        }
    }

    let phi_t = Tensor::new(phi, particles.device())?;
    let k_t = Tensor::new(k_mat, particles.device())?;
    Ok((phi_t, k_t))
}
