use candle_core::{Device, Result, safetensors::load};

use sml_candle::c06_smooth_and_physics_informed_ai::t12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/12_koopman_operators_sindy_equation_discovery_and_4dvar_data_assimilation.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Koopman operator forward
    let (psi, psi_next, x_next) = topic::koopman_forward(
        &w["x_state"],
        &w["w_enc"],
        &w["b_enc"],
        &w["k_mat"],
        &w["w_dec"],
        &w["b_dec"],
    )?;
    common::assert_max_diff(&psi, &w["psi"], 1e-5)?;
    common::assert_max_diff(&psi_next, &w["psi_next"], 1e-5)?;
    common::assert_max_diff(&x_next, &w["x_next"], 1e-5)?;

    // SINDy library
    let theta_x = topic::sindy_library(&w["x_state"])?;
    common::assert_max_diff(&theta_x, &w["theta_x"], 1e-5)?;

    // SINDy step
    let (x_dot_pred, xi_sparse) = topic::sindy_step(&w["theta_x"], &w["xi_noisy"], 0.05)?;
    common::assert_max_diff(&x_dot_pred, &w["x_dot_pred"], 1e-5)?;
    common::assert_max_diff(&xi_sparse, &w["xi_sparse"], 1e-5)?;

    // 4D-Var step
    let (x_analysis, delta_x) = topic::four_d_var_step(
        &w["x_b"],
        &w["b_cov_inv"],
        &w["y_obs"],
        &w["r_cov_inv"],
        &w["h_obs"],
        &w["m_dyn"],
    )?;
    common::assert_max_diff(&x_analysis, &w["x_analysis"], 1e-5)?;
    common::assert_max_diff(&delta_x, &w["delta_x"], 1e-5)?;

    Ok(())
}
