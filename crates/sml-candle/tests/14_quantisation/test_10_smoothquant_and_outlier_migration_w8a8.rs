use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t10_smoothquant_and_outlier_migration_w8a8 as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_smoothquant_and_outlier_migration_w8a8_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/03_llm_ptq_algorithms/03_smoothquant_and_outlier_migration_w8a8.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Smooth scale computation
    let scale = topic::compute_smoothquant_scale(&w["x_calib"], &w["w_orig"], 0.5)?;
    common::assert_max_diff(&scale, &w["scale"], 1e-5)?;

    // 3. Unsmoothed W8A8 output
    let x_slice = w["x_calib"].narrow(0, 0, 8)?;
    let out_unsmoothed = topic::w8a8_gemm_unfused(&x_slice, &w["w_orig"], Some(&w["b_orig"]))?;
    common::assert_max_diff(&out_unsmoothed, &w["out_unsmoothed"], 1e-5)?;

    // 4. SmoothQuant W8A8 output and intermediate states
    let (out_smooth, x_smooth, w_smooth) =
        topic::smoothquant_w8a8_gemm(&x_slice, &w["w_orig"], &scale, Some(&w["b_orig"]))?;
    common::assert_max_diff(&x_smooth, &w["x_smooth"], 1e-5)?;
    common::assert_max_diff(&w_smooth, &w["w_smooth"], 1e-5)?;
    common::assert_max_diff(&out_smooth, &w["out_smooth"], 1e-5)?;

    Ok(())
}
