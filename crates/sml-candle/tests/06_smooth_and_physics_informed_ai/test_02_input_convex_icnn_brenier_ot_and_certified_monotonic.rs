use candle_core::{safetensors::load, Device, Result};

use ai_candle::c06_smooth_and_physics_informed_ai::t02_input_convex_icnn_brenier_ot_and_certified_monotonic as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_input_convex_icnn_brenier_ot_and_certified_monotonic_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("06_smooth_and_physics_informed_ai/02_input_convex_icnn_brenier_ot_and_certified_monotonic.safetensors"), &device)?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let psi = topic::icnn_potential(
        &w["x_input"],
        &w["w_x0"],
        &w["b0"],
        &w["w_z1"],
        &w["w_x1"],
        &w["b1"],
    )?;
    common::assert_max_diff(&psi, &w["icnn_psi"], 1e-5)?;

    let t_map = topic::brenier_ot_gradient_map(
        &w["x_input"],
        &w["w_x0"],
        &w["b0"],
        &w["w_z1"],
        &w["w_x1"],
    )?;
    common::assert_max_diff(&t_map, &w["brenier_t_map"], 1e-5)?;

    Ok(())
}
