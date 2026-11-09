use candle_core::{safetensors::load, Device, Result};

use ai_candle::c15_compression_peft_and_speculative_inference::t08_speculative_decoding_lossless_rejection_sampling_and_drafting as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_speculative_decoding_lossless_rejection_sampling_and_drafting_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/03_speculative_decoding_lossless_rejection_sampling_and_drafting.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Speculative rejection step
    let (acc_mask, n_acc, res_p) = topic::speculative_rejection_step(
        &w["target_probs"],
        &w["draft_probs"],
        &w["draft_tokens"],
        &w["uniform_rands"],
    )?;
    common::assert_max_diff(&acc_mask, &w["accepted_mask"], 1e-5)?;
    common::assert_max_diff(&n_acc, &w["num_accepted"], 1e-5)?;
    common::assert_max_diff(&res_p, &w["residual_probs"], 1e-5)?;

    // 3. Prompt lookup draft
    let draft = topic::prompt_lookup_ngram_draft(&w["prompt_seq"], 3, 4)?;
    common::assert_max_diff(&draft, &w["draft_ngram"], 1e-5)?;

    Ok(())
}
