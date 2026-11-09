use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t13_omniquant_and_hqq_optimization_based_ptq as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_omniquant_and_hqq_optimization_based_ptq_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/03_llm_ptq_algorithms/06_omniquant_and_hqq_optimization_based_ptq.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. HQQ proximal quantization
    let (w_hqq, hqq_scale, hqq_zero) = topic::hqq_proximal_quantize(&w["w_orig"], 4)?;
    common::assert_max_diff(&w_hqq, &w["w_hqq"], 1e-5)?;
    common::assert_max_diff(&hqq_scale, &w["hqq_scale"], 1e-5)?;
    common::assert_max_diff(&hqq_zero, &w["hqq_zero"], 1e-5)?;

    let out_hqq = w["x_calib"].matmul(&w_hqq)?.broadcast_add(&w["b_orig"])?;
    common::assert_max_diff(&out_hqq, &w["out_hqq"], 1e-5)?;

    // 3. OmniQuant forward with optimized parameters
    let (out_omni, _) = topic::omniquant_forward(
        &w["x_calib"],
        &w["w_orig"],
        &w["gamma_opt"],
        &w["h_l_opt"],
        &w["h_u_opt"],
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_omni, &w["out_omniquant"], 1e-5)?;

    Ok(())
}
