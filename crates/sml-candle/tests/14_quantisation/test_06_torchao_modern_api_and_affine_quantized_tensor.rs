use candle_core::{Device, Result, safetensors::load};

use ai_candle::c14_quantisation::t06_torchao_modern_api_and_affine_quantized_tensor as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_torchao_modern_api_and_affine_quantized_tensor_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/02_pt2e_torchao_and_bitpacking/02_torchao_modern_api_and_affine_quantized_tensor.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Block-wise INT8 AQT
    let (w_deq, w_scales, w_int) =
        topic::affine_quantize_blockwise(&w["w_linear"], 16, -128.0, 127.0)?;
    common::assert_max_diff(&w_deq, &w["w_block_dequant"], 1e-5)?;
    common::assert_max_diff(&w_scales, &w["w_block_scales"], 1e-5)?;
    common::assert_max_diff(&w_int, &w["w_block_int"], 1e-5)?;

    let out_w = w["x_act"].matmul(&w_deq)?.broadcast_add(&w["b_linear"])?;
    common::assert_max_diff(&out_w, &w["out_w_only"], 1e-5)?;

    // 3. Dynamic W8A8 Linear
    let (y_w8a8, s_x, s_w) =
        topic::dynamic_w8a8_linear(&w["x_act"], &w["w_linear"], &w["b_linear"])?;
    common::assert_max_diff(&y_w8a8, &w["w8a8_out"], 1e-5)?;
    common::assert_max_diff(&s_x, &w["w8a8_s_x"], 1e-5)?;
    common::assert_max_diff(&s_w, &w["w8a8_s_w"], 1e-5)?;

    // 4. INT4 Weight-Only Linear
    let (out_w4, w4_deq, w4_scales) =
        topic::int4_weight_only_linear(&w["x_act"], &w["w_linear"], &w["b_linear"], 16)?;
    common::assert_max_diff(&out_w4, &w["out_w4_only"], 1e-5)?;
    common::assert_max_diff(&w4_deq, &w["w4_block_dequant"], 1e-5)?;
    common::assert_max_diff(&w4_scales, &w["w4_block_scales"], 1e-5)?;

    Ok(())
}
