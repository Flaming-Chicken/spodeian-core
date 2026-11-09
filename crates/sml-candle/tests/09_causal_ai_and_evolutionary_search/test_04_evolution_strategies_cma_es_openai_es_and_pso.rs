use candle_core::{safetensors::load, Device, Result};

use ai_candle::c09_causal_ai_and_evolutionary_search::t04_evolution_strategies_cma_es_openai_es_and_pso as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_evolution_strategies_cma_es_openai_es_and_pso_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("09_causal_ai_and_evolutionary_search/04_evolution_strategies_cma_es_openai_es_and_pso.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. OpenAI ES antithetic gradient
    let es_grad = topic::openai_es_gradient_estimate(
        &w["es_epsilons"],
        &w["es_pos_fit"],
        &w["es_neg_fit"],
        0.1,
    )?;
    common::assert_max_diff(&es_grad, &w["es_grad_ref"], 1e-5)?;

    // 3. CMA-ES mean and covariance update
    let (cma_mu, cma_cov) = topic::cma_es_mean_and_cov_step(
        &w["cma_mu_old"],
        &w["cma_cov_old"],
        &w["cma_pop"],
        &w["cma_weights"],
        0.2,
    )?;
    common::assert_max_diff(&cma_mu, &w["cma_mu_new_ref"], 1e-5)?;
    common::assert_max_diff(&cma_cov, &w["cma_cov_new_ref"], 1e-5)?;

    // 4. PSO velocity and position update
    let (pso_v, pso_x) = topic::pso_velocity_and_position_step(
        &w["pso_x"],
        &w["pso_v"],
        &w["pso_pbest"],
        &w["pso_gbest"],
        &w["pso_r1"],
        &w["pso_r2"],
        0.7,
        1.4,
        1.4,
    )?;
    common::assert_max_diff(&pso_v, &w["pso_v_next_ref"], 1e-5)?;
    common::assert_max_diff(&pso_x, &w["pso_x_next_ref"], 1e-5)?;

    Ok(())
}
