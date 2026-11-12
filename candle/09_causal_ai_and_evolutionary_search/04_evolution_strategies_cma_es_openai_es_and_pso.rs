//! Evolution Strategies CMA Es Openai Es and Pso
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// OpenAI Evolution Strategies antithetic gradient estimator.
pub fn openai_es_gradient_estimate(
    epsilons: &Tensor,
    pos_fitness: &Tensor,
    neg_fitness: &Tensor,
    sigma: f64,
) -> Result<Tensor> {
    let p = epsilons.dim(0)?;
    let diff = (pos_fitness - neg_fitness)?.unsqueeze(1)?;
    let weighted = epsilons.broadcast_mul(&diff)?;
    let sum_eps = weighted.sum(0)?;
    let scale = 1.0 / (2.0 * (p as f64) * sigma);
    sum_eps.affine(scale, 0.0)
}

/// CMA-ES elite recombination and covariance matrix update step.
pub fn cma_es_mean_and_cov_step(
    mu_old: &Tensor,
    cov_old: &Tensor,
    population: &Tensor,
    weights_elite: &Tensor,
    c_cov: f64,
) -> Result<(Tensor, Tensor)> {
    let w_col = weights_elite.unsqueeze(1)?;
    let mu_new = population.broadcast_mul(&w_col)?.sum(0)?;

    let y = population.broadcast_sub(&mu_old.unsqueeze(0)?)?;
    let w_y = y.broadcast_mul(&w_col)?;
    let c_mu = y.t()?.matmul(&w_y)?;

    let cov_term1 = cov_old.affine(1.0 - c_cov, 0.0)?;
    let cov_term2 = c_mu.affine(c_cov, 0.0)?;
    let cov_new = cov_term1.add(&cov_term2)?;

    Ok((mu_new, cov_new))
}

/// Particle Swarm Optimization (PSO) velocity and position update step.
pub fn pso_velocity_and_position_step(
    x: &Tensor,
    v: &Tensor,
    p_best: &Tensor,
    g_best: &Tensor,
    r1: &Tensor,
    r2: &Tensor,
    w_inertia: f64,
    c1: f64,
    c2: f64,
) -> Result<(Tensor, Tensor)> {
    let cog_diff = p_best.sub(x)?;
    let cognitive = r1.mul(&cog_diff)?.affine(c1, 0.0)?;

    let soc_diff = g_best.broadcast_sub(x)?;
    let social = r2.mul(&soc_diff)?.affine(c2, 0.0)?;

    let v_inertia = v.affine(w_inertia, 0.0)?;
    let v_next = v_inertia.add(&cognitive)?.add(&social)?;
    let x_next = x.add(&v_next)?;

    Ok((v_next, x_next))
}
