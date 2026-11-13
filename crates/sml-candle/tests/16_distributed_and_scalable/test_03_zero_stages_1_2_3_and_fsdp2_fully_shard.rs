use candle_core::{Device, Result, safetensors::load};

use ai_candle::c16_distributed_and_scalable::t03_zero_stages_1_2_3_and_fsdp2_fully_shard as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_zero_stages_1_2_3_and_fsdp2_fully_shard_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "16_distributed_and_scalable/03_zero_stages_1_2_3_and_fsdp2_fully_shard.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. FSDP2 Forward
    let (h, w_full) = topic::fsdp2_forward_matmul(&w["x"], &w["param_shards"])?;
    common::assert_max_diff(&w_full, &w["w_full"], 1e-5)?;
    common::assert_max_diff(&h, &w["fsdp_fwd_out"], 1e-5)?;

    // 2. FSDP2 Backward
    let (grad_w_full, grad_w_shards) =
        topic::fsdp2_backward_reducescatter(&w["x"], &w["grad_out"], 2)?;
    common::assert_max_diff(&grad_w_full, &w["grad_w_full"], 1e-5)?;
    common::assert_max_diff(&grad_w_shards, &w["grad_w_shards"], 1e-5)?;

    // 3. ZeRO-1 Adam Step
    let (p_new, m_new, v_new) = topic::zero1_adam_step(
        &w["p_init"],
        &w["g_init"],
        &w["m_init"],
        &w["v_init"],
        0.01,
        0.9,
        0.999,
        1e-8,
        1,
    )?;
    common::assert_max_diff(&p_new, &w["p_new"], 1e-5)?;
    common::assert_max_diff(&m_new, &w["m_new"], 1e-5)?;
    common::assert_max_diff(&v_new, &w["v_new"], 1e-5)?;

    Ok(())
}
