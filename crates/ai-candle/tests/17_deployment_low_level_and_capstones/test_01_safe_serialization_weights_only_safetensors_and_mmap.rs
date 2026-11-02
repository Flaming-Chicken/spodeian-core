use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t01_safe_serialization_weights_only_safetensors_and_mmap as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_safe_serialization_weights_only_safetensors_and_mmap_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("17_deployment_low_level_and_capstones/01_safe_serialization_weights_only_safetensors_and_mmap.safetensors"),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Safe loaded weights projection forward check
    let y_proj = topic::project_forward(&w["x_query"], &w["w_proj"], &w["b_proj"])?;
    common::assert_max_diff(&y_proj, &w["y_ref"], 1e-5)?;

    Ok(())
}
