use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t02_onnx_export_dynamo_and_onnxruntime_inference as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_onnx_export_dynamo_and_onnxruntime_inference_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("17_deployment_low_level_and_capstones/02_onnx_export_dynamo_and_onnxruntime_inference.safetensors"),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Dynamic shape 1: [1, 4, 4]
    let y1 = topic::onnx_dynamo_forward(&w["x1"], &w["w1"], &w["b1"], &w["w2"], &w["b2"])?;
    common::assert_max_diff(&y1, &w["y1"], 1e-5)?;

    // Dynamic shape 2: [2, 2, 4]
    let y2 = topic::onnx_dynamo_forward(&w["x2"], &w["w1"], &w["b1"], &w["w2"], &w["b2"])?;
    common::assert_max_diff(&y2, &w["y2"], 1e-5)?;

    Ok(())
}
