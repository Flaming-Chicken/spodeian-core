use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t23_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_23_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/04_transformers_and_heads/05_transformer_archetypes_bert_gpt_t5_vit_swin_perceiver_and_jamba.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Perceiver cross-attention bottleneck parity
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;
    let (perceiver_out, perceiver_attn) = topic::perceiver_cross_attention(
        &w["latents"],
        &w["inputs"],
        &w["w_q"],
        &w["w_k"],
        &w["w_v"],
        &w["w_o"],
        scale,
    )?;
    common::assert_max_diff(&perceiver_out, &w["perceiver_out"], 1e-5)?;
    common::assert_max_diff(&perceiver_attn, &w["perceiver_attn"], 1e-5)?;

    // 3. Causal decoder mask parity
    let causal_scores = topic::causal_decoder_mask(&w["raw_scores"])?;
    common::assert_max_diff(&causal_scores, &w["causal_scores"], 1e-5)?;

    Ok(())
}
