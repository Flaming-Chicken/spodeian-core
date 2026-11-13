use candle_core::{Device, Result, safetensors::load};

use sml_candle::c06_smooth_and_physics_informed_ai::t07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/07_numerical_integration_pcec_rk45_neural_ode_ltc_and_sde.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Numerical Integrators
    let (f0, x_euler, x_heun, x_pcec, x_rk4) =
        topic::integrators_step(&w["x0"], &w["w1"], &w["b1"], &w["w2"], &w["b2"], 0.1)?;
    common::assert_max_diff(&f0, &w["f0"], 1e-5)?;
    common::assert_max_diff(&x_euler, &w["x_euler"], 1e-5)?;
    common::assert_max_diff(&x_heun, &w["x_heun"], 1e-5)?;
    common::assert_max_diff(&x_pcec, &w["x_pcec"], 1e-5)?;
    common::assert_max_diff(&x_rk4, &w["x_rk4"], 1e-5)?;

    // Adjoint sensitivity
    let j0 = topic::eval_jacobian(&w["x0"], &w["w1"], &w["b1"], &w["w2"])?;
    common::assert_max_diff(&j0, &w["j0"], 1e-5)?;

    let adj_0 = topic::adjoint_step(&w["adj_T"], &j0, 0.1)?;
    common::assert_max_diff(&adj_0, &w["adj_0"], 1e-5)?;

    // CfC network
    let x_cfc = topic::cfc_step(&w["x0"], &f0, &w["w_cfc1"], &w["w_cfc2"], 0.1)?;
    common::assert_max_diff(&x_cfc, &w["x_cfc"], 1e-5)?;

    // Neural SDE
    let (x_em, x_milstein) = topic::sde_step(&w["x0"], &f0, &w["dW"], 0.2, 0.1)?;
    common::assert_max_diff(&x_em, &w["x_em"], 1e-5)?;
    common::assert_max_diff(&x_milstein, &w["x_milstein"], 1e-5)?;

    Ok(())
}
