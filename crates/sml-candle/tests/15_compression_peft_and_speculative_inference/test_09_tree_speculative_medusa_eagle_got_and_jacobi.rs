use candle_core::{safetensors::load, Device, Result};

use ai_candle::c15_compression_peft_and_speculative_inference::t09_tree_speculative_medusa_eagle_got_and_jacobi as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_tree_speculative_medusa_eagle_got_and_jacobi_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/04_tree_speculative_medusa_eagle_got_and_jacobi.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Medusa heads
    let (m_logits, t_tokens) =
        topic::medusa_heads_predict(&w["hidden"], &w["head_weights"], &w["head_biases"])?;
    common::assert_max_diff(&m_logits, &w["medusa_logits"], 1e-5)?;
    common::assert_max_diff(&t_tokens, &w["top_tokens"], 1e-5)?;

    // 3. Tree mask
    let t_mask = topic::build_tree_attention_mask(&w["parent_indices"])?;
    common::assert_max_diff(&t_mask, &w["tree_mask"], 1e-5)?;

    // 4. Longest path
    let (p_mask, p_len) = topic::tree_longest_accepted_path(
        &w["candidate_tokens"],
        &w["target_tokens"],
        &w["parent_indices"],
    )?;
    common::assert_max_diff(&p_mask, &w["path_mask"], 1e-5)?;
    common::assert_max_diff(&p_len, &w["path_length"], 1e-5)?;

    // 5. Jacobi step
    let (x_nxt, j_res) =
        topic::jacobi_lookahead_step(&w["x_jacobi"], &w["w_jacobi"], &w["b_jacobi"])?;
    common::assert_max_diff(&x_nxt, &w["x_next"], 1e-5)?;
    common::assert_max_diff(&j_res, &w["jacobi_res"], 1e-5)?;

    Ok(())
}
