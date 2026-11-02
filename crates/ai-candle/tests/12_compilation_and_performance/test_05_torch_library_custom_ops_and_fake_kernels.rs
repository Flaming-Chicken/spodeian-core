use candle_core::{safetensors::load, Device, Result};

use ai_candle::c12_compilation_and_performance::t05_torch_library_custom_ops_and_fake_kernels as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_torch_library_custom_ops_and_fake_kernels_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("12_compilation_and_performance/05_torch_library_custom_ops_and_fake_kernels.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Soft shrink
    let shrink = topic::custom_soft_shrink(&w["shrink_x"], 0.5)?;
    common::assert_max_diff(&shrink, &w["shrink_ref"], 1e-5)?;

    // 3. Asymmetric quantize
    let (x_hat, scale, zero_point) = topic::custom_asymmetric_quantize(&w["quant_x"], 1e-7)?;
    common::assert_max_diff(&x_hat, &w["quant_x_hat"], 1e-5)?;
    common::assert_max_diff(&scale, &w["quant_scale"], 1e-5)?;
    common::assert_max_diff(&zero_point, &w["quant_zero"], 1e-5)?;

    // 4. Depthwise Conv1D
    let conv = topic::custom_depthwise_conv1d(&w["conv_x"], &w["conv_w"], &w["conv_b"])?;
    common::assert_max_diff(&conv, &w["conv_ref"], 1e-5)?;

    Ok(())
}
