use candle_core::{Device, Result, safetensors::load};

use sml_candle::c09_causal_ai_and_evolutionary_search::t01_causal_discovery_notears_scm_and_instrumental_variables as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_causal_discovery_notears_scm_and_instrumental_variables_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "09_causal_ai_and_evolutionary_search/01_causal_discovery_notears_scm_and_instrumental_variables.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. NOTEARS acyclicity constraint
    let (h, grad_h) = topic::notears_acyclicity_constraint(&w["w_dag"])?;
    common::assert_max_diff(&h, &w["h_dag_ref"], 1e-5)?;
    common::assert_max_diff(&grad_h, &w["grad_h_ref"], 1e-5)?;

    // 3. SCM forward and counterfactual
    let (fact, cf) = topic::scm_generate_and_counterfactual(&w["scm_w"], &w["scm_u"], 1, 3.0)?;
    common::assert_max_diff(&fact, &w["scm_x_factual_ref"], 1e-5)?;
    common::assert_max_diff(&cf, &w["counterfactual_x_ref"], 1e-5)?;

    // 4. Two-Stage Least Squares (2SLS)
    let (t_hat, beta) =
        topic::two_stage_least_squares(&w["z_inst"], &w["t_treat"], &w["y_outcome"])?;
    common::assert_max_diff(&t_hat, &w["t_hat_ref"], 1e-5)?;
    common::assert_max_diff(&beta, &w["beta_2sls_ref"], 1e-5)?;

    Ok(())
}
