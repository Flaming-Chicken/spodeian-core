use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t07_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/02_pt2e_torchao_and_bitpacking/03_sub_byte_bitpacking_tmac_lut_and_gguf_k_quants.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. 2x4-bit packing
    let p2x4 = topic::bitpack_2x4bit(&w["raw_nibbles"])?;
    common::assert_max_diff(&p2x4, &w["packed_2x4bit"], 1e-5)?;
    let unp2x4 = topic::bitunpack_2x4bit(&w["packed_2x4bit"])?;
    common::assert_max_diff(&unp2x4, &w["unpacked_nibbles"], 1e-5)?;

    // 3. 5x ternary packing
    let p_trit = topic::bitpack_5x_ternary(&w["raw_trits"])?;
    common::assert_max_diff(&p_trit, &w["packed_ternary"], 1e-5)?;
    let unp_trit = topic::bitunpack_5x_ternary(&w["packed_ternary"])?;
    common::assert_max_diff(&unp_trit, &w["unpacked_trits"], 1e-5)?;

    // 4. T-MAC LUT
    let lut = topic::tmac_precompute_lut(&w["tmac_act"])?;
    common::assert_max_diff(&lut, &w["tmac_lut"], 1e-5)?;
    let tmac_out = topic::tmac_lut_lookup(&lut, &w["tmac_weight_indices"])?;
    common::assert_max_diff(&tmac_out, &w["tmac_dot_ref"], 1e-5)?;

    // 5. GGUF Q4_K
    let (deq, sup_s, sub_s) = topic::gguf_q4_k_quantize_block(&w["gguf_raw_w"])?;
    common::assert_max_diff(&deq, &w["gguf_dequant"], 1e-5)?;
    common::assert_max_diff(&sup_s, &w["gguf_super_scale"], 1e-5)?;
    common::assert_max_diff(&sub_s, &w["gguf_sub_scales"], 1e-5)?;

    Ok(())
}
