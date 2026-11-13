use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t08_rtn_vs_gptq_hessian_error_compensation as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_rtn_vs_gptq_hessian_error_compensation_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/03_llm_ptq_algorithms/01_rtn_vs_gptq_hessian_error_compensation.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Hessian and Damped Inverse
    let (h, h_inv) = topic::compute_damped_hessian_inv(&w["x_calib"], 0.01)?;
    common::assert_max_diff(&h, &w["h_matrix"], 1e-5)?;
    common::assert_max_diff(&h_inv, &w["h_inv"], 1e-4)?;

    // 3. RTN Quantization
    let w_rtn = topic::rtn_quantize(&w["w_orig"], -8.0, 7.0)?;
    common::assert_max_diff(&w_rtn, &w["w_rtn"], 1e-5)?;

    // 4. Standard GPTQ
    let w_gptq = topic::gptq_quantize(&w["w_orig"], &w["h_inv"], -8.0, 7.0)?;
    common::assert_max_diff(&w_gptq, &w["w_gptq"], 1e-5)?;

    // 5. GPTQ with Act-Order
    let (w_act_gptq, perm, _) =
        topic::gptq_with_act_order(&w["x_calib"], &w["w_orig"], 0.01, -8.0, 7.0)?;
    common::assert_max_diff(&w_act_gptq, &w["w_act_gptq"], 1e-5)?;
    common::assert_max_diff(&perm, &w["perm_indices"], 1e-5)?;

    Ok(())
}
