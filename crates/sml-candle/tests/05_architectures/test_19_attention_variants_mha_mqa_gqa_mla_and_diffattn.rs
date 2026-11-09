use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t19_attention_variants_mha_mqa_gqa_mla_and_diffattn as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_19_attention_variants_mha_mqa_gqa_mla_and_diffattn_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. GQA parity
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;
    let (gqa_out, gqa_weights) = topic::grouped_query_attention(&w["q"], &w["k"], &w["v"], scale)?;
    common::assert_max_diff(&gqa_out, &w["gqa_out"], 1e-5)?;
    common::assert_max_diff(&gqa_weights, &w["gqa_weights"], 1e-5)?;

    // 3. DiffAttn parity
    let lambda_val = w["lambda_val"].to_vec1::<f32>()?[0] as f64;
    let (diff_out, diff_weights) = topic::differential_attention(
        &w["q1"],
        &w["k1"],
        &w["q2"],
        &w["k2"],
        &w["v_diff"],
        lambda_val,
        scale,
    )?;
    common::assert_max_diff(&diff_out, &w["diff_out"], 1e-5)?;
    common::assert_max_diff(&diff_weights, &w["diff_weights"], 1e-5)?;

    Ok(())
}
