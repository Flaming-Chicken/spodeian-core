use candle_core::{safetensors::load, Device, Result};

use ai_candle::c06_smooth_and_physics_informed_ai::t04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/04_physics_informed_pinns_deep_ritz_hard_bc_and_ntk_balancing.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Forward and derivatives
    let (u_raw, u_x, u_xx, phi) = topic::pinn_forward_and_derivatives(
        &w["x_colloc"],
        &w["w1"],
        &w["b1"],
        &w["w2"],
        &w["b2"],
    )?;
    common::assert_max_diff(&u_raw, &w["u_raw"], 1e-5)?;
    common::assert_max_diff(&u_x, &w["u_x"], 1e-5)?;
    common::assert_max_diff(&u_xx, &w["u_xx"], 1e-5)?;

    // PDE residual and loss
    let (pde_residual, loss_pde) = topic::pde_residual_and_loss(&w["x_colloc"], &u_xx)?;
    common::assert_max_diff(&pde_residual, &w["pde_residual"], 1e-5)?;
    common::assert_max_diff(&loss_pde, &w["loss_pde"], 1e-5)?;

    // Hard Boundary Conditions
    let (u_hard, u_hard_x) = topic::hard_boundary_conditions(&w["x_colloc"], &u_raw, &u_x)?;
    common::assert_max_diff(&u_hard, &w["u_hard"], 1e-5)?;
    common::assert_max_diff(&u_hard_x, &w["u_hard_x"], 1e-5)?;

    // Deep Ritz Energy
    let ritz_energy = topic::deep_ritz_energy(&w["x_colloc"], &u_hard, &u_hard_x)?;
    common::assert_max_diff(&ritz_energy, &w["ritz_energy"], 1e-5)?;

    // NTK Balancing
    let (loss_bc, g_pde, g_bc, lambda_ntk, g_balanced) = topic::ntk_loss_and_balancing(
        &w["x_colloc"],
        &phi,
        &pde_residual,
        &w["w1"],
        &w["b1"],
        &w["w2"],
        &w["b2"],
    )?;
    common::assert_max_diff(&loss_bc, &w["loss_bc"], 1e-5)?;
    common::assert_max_diff(&g_pde, &w["g_pde"], 1e-5)?;
    common::assert_max_diff(&g_bc, &w["g_bc"], 1e-5)?;
    common::assert_max_diff(&lambda_ntk, &w["lambda_ntk"], 5e-5)?;
    common::assert_max_diff(&g_balanced, &w["g_balanced"], 1e-5)?;

    Ok(())
}
