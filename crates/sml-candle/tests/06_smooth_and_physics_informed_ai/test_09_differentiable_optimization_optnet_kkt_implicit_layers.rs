use candle_core::{Device, Result, safetensors::load};

use sml_candle::c06_smooth_and_physics_informed_ai::t09_differentiable_optimization_optnet_kkt_implicit_layers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_differentiable_optimization_optnet_kkt_implicit_layers_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/09_differentiable_optimization_optnet_kkt_implicit_layers.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Equality QP
    let (z_eq, nu_eq) = topic::solve_equality_qp(&w["Q"], &w["p"], &w["A"], &w["b"])?;
    common::assert_max_diff(&z_eq, &w["z_eq"], 1e-5)?;
    common::assert_max_diff(&nu_eq, &w["nu_eq"], 1e-5)?;

    // Implicit KKT backward step
    let (d_z, grad_p, grad_b) = topic::optnet_backward_step(&w["Q"], &w["A"], &w["grad_loss_z"])?;
    common::assert_max_diff(&d_z, &w["d_z"], 1e-5)?;
    common::assert_max_diff(&grad_p, &w["grad_p"], 1e-5)?;
    common::assert_max_diff(&grad_b, &w["grad_b"], 1e-5)?;

    // Log-barrier inequality QP
    let (z_barrier, lambda_barrier) =
        topic::barrier_qp_solve(&w["Q"], &w["p"], &w["G"], &w["h_bound"], 20, 0.08, 5.0)?;
    common::assert_max_diff(&z_barrier, &w["z_barrier"], 1e-5)?;
    common::assert_max_diff(&lambda_barrier, &w["lambda_barrier"], 1e-5)?;

    Ok(())
}
