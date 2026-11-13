use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t21_flash_attention_online_softmax_and_flex_attention as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_21_flash_attention_online_softmax_and_flex_attention_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/04_transformers_and_heads/03_flash_attention_online_softmax_and_flex_attention.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. FlashAttention online softmax parity
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;
    let block_size = w["block_size"].to_vec1::<i64>()?[0] as usize;
    let (flash_out, _m, _l) =
        topic::online_softmax_flash_attention(&w["q"], &w["k"], &w["v"], block_size, scale)?;
    common::assert_max_diff(&flash_out, &w["flash_out"], 1e-5)?;
    common::assert_max_diff(&flash_out, &w["std_out"], 1e-5)?;

    Ok(())
}
