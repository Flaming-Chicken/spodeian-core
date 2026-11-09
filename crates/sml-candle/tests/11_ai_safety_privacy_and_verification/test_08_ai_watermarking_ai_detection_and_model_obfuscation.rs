use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t08_ai_watermarking_ai_detection_and_model_obfuscation as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_ai_watermarking_ai_detection_and_model_obfuscation_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/08_ai_watermarking_ai_detection_and_model_obfuscation.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Watermark logit bias
    let w_logits = topic::watermark_logit_bias(&w["logits"], &w["green_mask"], 2.5)?;
    common::assert_max_diff(&w_logits, &w["watermarked_logits_ref"], 1e-5)?;

    // 3. Watermark z-score
    let z_scores = topic::watermark_z_score(&w["green_counts"], &w["seq_lens"], 0.5)?;
    common::assert_max_diff(&z_scores, &w["z_scores_ref"], 1e-5)?;

    // 4. Fast-DetectGPT curvature score
    let detect_scores = topic::fast_detect_gpt_score(&w["log_probs"], &w["ref_log_probs"])?;
    common::assert_max_diff(&detect_scores, &w["detect_scores_ref"], 1e-5)?;

    // 5. Passport layer
    let pass_out = topic::passport_layer_forward(
        &w["pass_x"],
        &w["pass_w"],
        &w["pass_b"],
        &w["passport_key"],
        &w["passport_weight"],
    )?;
    common::assert_max_diff(&pass_out, &w["passport_out_ref"], 1e-5)?;

    Ok(())
}
