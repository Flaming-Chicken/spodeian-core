use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t11_llm_int8_mixed_precision_outlier_decomposition as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_llm_int8_mixed_precision_outlier_decomposition_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/03_llm_ptq_algorithms/04_llm_int8_mixed_precision_outlier_decomposition.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Pure INT8 baseline
    let (deq_x, _) = topic::quantize_row_int8(&w["x_calib"])?;
    let (deq_w, _) = topic::quantize_col_int8(&w["w_orig"])?;
    let out_pure = deq_x.matmul(&deq_w)?.broadcast_add(&w["b_orig"])?;
    common::assert_max_diff(&out_pure, &w["out_pure_int8"], 1e-5)?;

    // 3. LLM.int8 decomposition
    let (out_total, out_fp, out_int8, outlier_indices) =
        topic::llm_int8_linear(&w["x_calib"], &w["w_orig"], Some(&w["b_orig"]), 6.0)?;
    common::assert_max_diff(&out_total, &w["out_total"], 1e-5)?;
    common::assert_max_diff(&out_fp, &w["out_fp"], 1e-5)?;
    common::assert_max_diff(&out_int8, &w["out_int8"], 1e-5)?;
    common::assert_max_diff(&outlier_indices, &w["outlier_indices"], 1e-5)?;

    Ok(())
}
