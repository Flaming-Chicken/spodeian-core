use candle_core::{Device, Result, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t07_kv_cache_paged_radix_h2o_snapkv_and_yoco as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_kv_cache_paged_radix_h2o_snapkv_and_yoco_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/02_kv_cache_paged_radix_h2o_snapkv_and_yoco.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Paged Attention
    let (p_out, p_weights) = topic::paged_kv_gather_and_attend(
        &w["query"],
        &w["key_blocks"],
        &w["value_blocks"],
        &w["block_table"],
        10,
        4,
    )?;
    common::assert_max_diff(&p_out, &w["paged_out"], 1e-5)?;
    common::assert_max_diff(&p_weights, &w["paged_weights"], 1e-5)?;

    // 3. H2O Cache Eviction
    let h_mask = topic::h2o_evict_mask(&w["cumulative_scores"], 4, 3)?;
    common::assert_max_diff(&h_mask, &w["h2o_mask"], 1e-5)?;

    // 4. SnapKV
    let s_mask = topic::snapkv_select_mask(&w["prompt_attn"], 4, 6)?;
    common::assert_max_diff(&s_mask, &w["snapkv_mask"], 1e-5)?;

    // 5. YOCO
    let y_out = topic::yoco_cross_layer_forward(
        &w["h_mid"],
        &w["k_shared"],
        &w["v_shared"],
        &w["w_q"],
        &w["w_out"],
    )?;
    common::assert_max_diff(&y_out, &w["yoco_out"], 1e-5)?;

    Ok(())
}
