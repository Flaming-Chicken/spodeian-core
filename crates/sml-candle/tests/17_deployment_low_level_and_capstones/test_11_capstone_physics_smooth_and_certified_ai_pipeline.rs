use candle_core::{Device, Result, safetensors::load};

use sml_candle::c17_deployment_low_level_and_capstones::t11_capstone_physics_smooth_and_certified_ai_pipeline as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_capstone_physics_smooth_and_certified_ai_pipeline_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/11_capstone_physics_smooth_and_certified_ai_pipeline.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let (y_smooth, q_next, p_next, e_init, e_next) =
        topic::physics_certified_pipeline(&w["q"], &w["p"], &w["w_v"], &w["w_l"], &w["b_l"], 0.05)?;

    common::assert_max_diff(&y_smooth, &w["y_smooth"], 1e-5)?;
    common::assert_max_diff(&q_next, &w["q_next"], 1e-5)?;
    common::assert_max_diff(&p_next, &w["p_next"], 1e-5)?;
    common::assert_max_diff(&e_init, &w["energy_init"], 1e-5)?;
    common::assert_max_diff(&e_next, &w["energy_next"], 1e-5)?;

    Ok(())
}
