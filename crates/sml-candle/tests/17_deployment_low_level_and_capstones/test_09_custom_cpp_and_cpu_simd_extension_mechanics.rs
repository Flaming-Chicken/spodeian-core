use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t09_custom_cpp_and_cpu_simd_extension_mechanics as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_custom_cpp_and_cpu_simd_extension_mechanics_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/09_custom_cpp_and_cpu_simd_extension_mechanics.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let y = topic::fused_simd_fma_leaky_relu(&w["x"], &w["w"], &w["b"], 0.1)?;
    common::assert_max_diff(&y, &w["y"], 1e-5)?;

    Ok(())
}
