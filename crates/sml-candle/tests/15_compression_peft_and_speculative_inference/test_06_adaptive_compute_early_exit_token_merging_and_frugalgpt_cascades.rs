use candle_core::{Device, Result, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t06_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/01_adaptive_compute_early_exit_token_merging_and_frugalgpt_cascades.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Early-exit prediction
    let (probs, conf, exit_mask) =
        topic::early_exit_predict(&w["hidden"], &w["head_w"], &w["head_b"], 0.6)?;
    common::assert_max_diff(&probs, &w["probs"], 1e-5)?;
    common::assert_max_diff(&conf, &w["conf"], 1e-5)?;
    common::assert_max_diff(&exit_mask, &w["exit_mask"], 1e-5)?;

    // 3. Token Merging (ToMe)
    let (x_merged, merged_sizes) =
        topic::tome_bipartite_merge(&w["x_tokens"], 2, &w["init_sizes"])?;
    common::assert_max_diff(&x_merged, &w["x_merged"], 1e-5)?;
    common::assert_max_diff(&merged_sizes, &w["merged_sizes"], 1e-5)?;

    // 4. FrugalGPT Cascades
    let (routed_idx, total_cost) =
        topic::frugal_cascade_route(&w["confidences"], &w["costs"], &w["thresholds"])?;
    common::assert_max_diff(&routed_idx, &w["routed_idx"], 1e-5)?;
    common::assert_max_diff(&total_cost, &w["total_cost"], 1e-5)?;

    Ok(())
}
