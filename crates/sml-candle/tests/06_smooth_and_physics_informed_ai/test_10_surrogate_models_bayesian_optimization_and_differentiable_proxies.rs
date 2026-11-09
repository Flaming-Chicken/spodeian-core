use candle_core::{safetensors::load, Device, Result};

use ai_candle::c06_smooth_and_physics_informed_ai::t10_surrogate_models_bayesian_optimization_and_differentiable_proxies as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_surrogate_models_bayesian_optimization_and_differentiable_proxies_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(common::get_weights_path("06_smooth_and_physics_informed_ai/10_surrogate_models_bayesian_optimization_and_differentiable_proxies.safetensors"), &device)?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // RBF kernel matrix
    let k_sy = topic::rbf_kernel(
        &w["x_test"],
        &w["x_train"],
        &w["lengthscale"],
        &w["variance"],
    )?;
    common::assert_max_diff(&k_sy, &w["k_sy"], 1e-5)?;

    // GP posterior
    let (mu_s, var_s) = topic::gp_posterior(
        &w["x_train"],
        &w["y_train"],
        &w["x_test"],
        &w["lengthscale"],
        &w["variance"],
        &w["noise_var"],
    )?;
    common::assert_max_diff(&mu_s, &w["mu_s"], 1e-5)?;
    common::assert_max_diff(&var_s, &w["var_s"], 1e-5)?;

    // Expected Improvement acquisition
    let ei = topic::expected_improvement(&w["mu_s"], &w["var_s"], &w["best_y"], 0.01)?;
    common::assert_max_diff(&ei, &w["ei_acq"], 1e-5)?;

    // UCB acquisition
    let ucb = topic::ucb_acquisition(&w["mu_s"], &w["var_s"], 2.0)?;
    common::assert_max_diff(&ucb, &w["ucb_acq"], 1e-5)?;

    Ok(())
}
