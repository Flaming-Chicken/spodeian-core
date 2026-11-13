use candle_core::{Device, Result, safetensors::load};

use ai_candle::c16_distributed_and_scalable::t04_tensor_parallelism_megatron_sequence_and_async_tp as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_tensor_parallelism_megatron_sequence_and_async_tp_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "16_distributed_and_scalable/04_tensor_parallelism_megatron_sequence_and_async_tp.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Megatron TP MLP
    let (tp_out, partials) = topic::megatron_tp_mlp(&w["x"], &w["w1_shards"], &w["w2_shards"])?;
    common::assert_max_diff(&partials, &w["partial_shards"], 1e-5)?;
    common::assert_max_diff(&tp_out, &w["tp_mlp_out"], 1e-5)?;

    // 2. Sequence Parallelism MLP
    let (sp_out, x_full) =
        topic::sequence_parallel_mlp(&w["x_seq_shards"], &w["w1_shards"], &w["w2_shards"])?;
    common::assert_max_diff(&x_full, &w["x_full"], 1e-5)?;
    common::assert_max_diff(&sp_out, &w["sp_mlp_out_shards"], 1e-5)?;

    Ok(())
}
