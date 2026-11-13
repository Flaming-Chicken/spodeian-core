use candle_core::{Device, Result, safetensors::load};

use ai_candle::c06_smooth_and_physics_informed_ai::t08_deep_equilibrium_models_deq_implicit_differentiation as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_deep_equilibrium_models_deq_implicit_differentiation_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Picard iteration
    let (z_pic, res_pic) =
        topic::fixed_point_iteration(&w["x_input"], &w["w_z"], &w["w_x"], &w["b_z"], 20)?;
    common::assert_max_diff(&z_pic, &w["z_picard"], 1e-5)?;
    common::assert_max_diff(&res_pic, &w["res_picard"], 1e-5)?;

    // Anderson acceleration
    let (z_and, res_and) =
        topic::anderson_acceleration(&w["x_input"], &w["w_z"], &w["w_x"], &w["b_z"], 2, 15, 1.0)?;
    common::assert_max_diff(&z_and, &w["z_anderson"], 1e-5)?;
    common::assert_max_diff(&res_and, &w["res_anderson"], 1e-5)?;

    // Implicit differentiation adjoint solve
    let (adj_v, grad_x, grad_wz) = topic::implicit_jacobian_transpose_solve(
        &w["z_picard"],
        &w["x_input"],
        &w["w_z"],
        &w["w_x"],
        &w["b_z"],
        &w["grad_loss_z"],
        20,
    )?;
    common::assert_max_diff(&adj_v, &w["adj_v"], 1e-5)?;
    common::assert_max_diff(&grad_x, &w["grad_x"], 1e-5)?;
    common::assert_max_diff(&grad_wz, &w["grad_wz"], 1e-5)?;

    // Monotone operator splitting
    let z_mon = topic::mon_operator_splitting(
        &w["x_input"],
        &w["w_sym"],
        &w["w_skew"],
        &w["w_mon_x"],
        &w["b_mon"],
        0.2,
        15,
    )?;
    common::assert_max_diff(&z_mon, &w["z_mon"], 1e-5)?;

    Ok(())
}
