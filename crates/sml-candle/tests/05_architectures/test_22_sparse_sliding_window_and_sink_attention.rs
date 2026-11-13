use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t22_sparse_sliding_window_and_sink_attention as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_22_sparse_sliding_window_and_sink_attention_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/04_transformers_and_heads/04_sparse_sliding_window_and_sink_attention.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Sink attention parity
    let num_sinks = w["num_sinks"].to_vec1::<i64>()?[0] as usize;
    let window_size = w["window_size"].to_vec1::<i64>()?[0] as usize;
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;

    let (out, weights) =
        topic::streaming_sink_attention(&w["q"], &w["k"], &w["v"], num_sinks, window_size, scale)?;
    common::assert_max_diff(&out, &w["sink_out"], 1e-5)?;
    common::assert_max_diff(&weights, &w["sink_weights"], 1e-5)?;

    Ok(())
}
