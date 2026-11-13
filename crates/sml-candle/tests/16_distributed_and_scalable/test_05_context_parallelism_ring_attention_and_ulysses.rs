use candle_core::{Device, Result, safetensors::load};

use sml_candle::c16_distributed_and_scalable::t05_context_parallelism_ring_attention_and_ulysses as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_context_parallelism_ring_attention_and_ulysses_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "16_distributed_and_scalable/05_context_parallelism_ring_attention_and_ulysses.safetensors",
        ),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Standard attention reference
    let ref_attn = topic::standard_attention(&w["q_full"], &w["k_full"], &w["v_full"])?;
    common::assert_max_diff(&ref_attn, &w["attn_full_ref"], 1e-5)?;

    // Ring Attention
    let (ring_shards, _) =
        topic::ring_attention_forward(&w["q_shards"], &w["k_shards"], &w["v_shards"])?;
    common::assert_max_diff(&ring_shards, &w["ring_attn_out_shards"], 1e-5)?;

    // DeepSpeed Ulysses
    let (ulysses_shards, _) =
        topic::deepspeed_ulysses_forward(&w["q_shards"], &w["k_shards"], &w["v_shards"])?;
    common::assert_max_diff(&ulysses_shards, &w["ulysses_attn_out_shards"], 1e-5)?;

    Ok(())
}
