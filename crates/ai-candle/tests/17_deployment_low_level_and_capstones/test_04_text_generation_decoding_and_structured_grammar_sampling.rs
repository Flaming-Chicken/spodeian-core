use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t04_text_generation_decoding_and_structured_grammar_sampling as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_text_generation_decoding_and_structured_grammar_sampling_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/04_text_generation_decoding_and_structured_grammar_sampling.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let rep = topic::apply_repetition_penalty(&w["logits"], &w["history"], 1.25)?;
    common::assert_max_diff(&rep, &w["rep_penalized"], 1e-5)?;

    let masked = topic::apply_grammar_mask(&rep, &w["grammar_mask"], topic::MASK_VALUE)?;
    common::assert_max_diff(&masked, &w["masked_logits"], 1e-5)?;

    let temp = topic::apply_temperature(&masked, 0.8)?;
    common::assert_max_diff(&temp, &w["temp_logits"], 1e-5)?;

    let topk = topic::apply_top_k(&temp, 4, topic::MASK_VALUE)?;
    common::assert_max_diff(&topk, &w["top_k_logits"], 1e-5)?;

    let (probs, greedy) = topic::decoding_pipeline(
        &w["logits"],
        &w["history"],
        &w["grammar_mask"],
        0.8,
        1.25,
        4,
        0.1,
    )?;
    common::assert_max_diff(&probs, &w["probs"], 1e-5)?;
    common::assert_max_diff(&greedy, &w["greedy_tokens"], 1e-5)?;

    Ok(())
}
