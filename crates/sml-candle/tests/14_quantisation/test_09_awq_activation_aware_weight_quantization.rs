use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t09_awq_activation_aware_weight_quantization as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_awq_activation_aware_weight_quantization_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/03_llm_ptq_algorithms/02_awq_activation_aware_weight_quantization.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. RTN baseline
    let (w_rtn, _) = topic::rtn_quantize_per_column(&w["w_orig"], -8.0, 7.0)?;
    common::assert_max_diff(&w_rtn, &w["w_rtn"], 1e-5)?;

    // 3. AWQ optimal search
    let alpha_grid = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0];
    let (best_scale, best_alpha, w_awq) =
        topic::awq_search_optimal_scale(&w["x_calib"], &w["w_orig"], &alpha_grid, -8.0, 7.0)?;

    let expected_alpha = w["best_alpha"].to_vec1::<f32>()?[0];
    assert!((best_alpha - expected_alpha).abs() < 1e-5);
    common::assert_max_diff(&best_scale, &w["best_scale"], 1e-5)?;
    common::assert_max_diff(&w_awq, &w["w_awq"], 1e-5)?;

    // 4. Output activation parity
    let x_slice = w["x_calib"].narrow(0, 0, 8)?;
    let out_awq = x_slice.matmul(&w_awq)?.broadcast_add(&w["b_orig"])?;
    common::assert_max_diff(&out_awq, &w["out_awq"], 1e-5)?;

    Ok(())
}
