use candle_core::{Device, Result, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t14_peft_adapters_ia3_prefix_prompt_and_llama_adapter as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_14_peft_adapters_ia3_prefix_prompt_and_llama_adapter_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/04_peft_adapters_ia3_prefix_prompt_and_llama_adapter.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. (IA)^3
    let ia3_out = topic::ia3_rescale(&w["v"], &w["ia3_scale"])?;
    common::assert_max_diff(&ia3_out, &w["ia3_out"], 1e-5)?;

    // 2. Prefix Attention
    let prefix_out =
        topic::prefix_attention(&w["q"], &w["k"], &w["v"], &w["prefix_k"], &w["prefix_v"])?;
    common::assert_max_diff(&prefix_out, &w["prefix_attn_out"], 1e-5)?;

    // 3. LLaMA-Adapter Zero-Gate
    let llama_out = topic::llama_adapter_forward(
        &w["q"],
        &w["k"],
        &w["v"],
        &w["prefix_k"],
        &w["prefix_v"],
        &w["gate"],
    )?;
    common::assert_max_diff(&llama_out, &w["llama_adapter_out"], 1e-5)?;

    Ok(())
}
