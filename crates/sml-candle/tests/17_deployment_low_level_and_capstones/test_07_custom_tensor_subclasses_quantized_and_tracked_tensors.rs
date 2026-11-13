use candle_core::{Device, Result, safetensors::load};

use ai_candle::c17_deployment_low_level_and_capstones::t07_custom_tensor_subclasses_quantized_and_tracked_tensors as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_custom_tensor_subclasses_quantized_and_tracked_tensors_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/07_custom_tensor_subclasses_quantized_and_tracked_tensors.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let dequant_w = topic::dequantize_affine(&w["int_w"], &w["scale"], &w["zero_point"])?;
    common::assert_max_diff(&dequant_w, &w["w"], 1e-5)?;

    let (y_primal, y_tangent) =
        topic::forward_dual_affine(&w["x_primal"], &w["x_tangent"], &dequant_w, &w["b"])?;
    common::assert_max_diff(&y_primal, &w["y_primal"], 1e-5)?;
    common::assert_max_diff(&y_tangent, &w["y_tangent"], 1e-5)?;

    Ok(())
}
