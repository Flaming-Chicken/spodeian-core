use candle_core::{Device, Result, safetensors::load};

use ai_candle::c16_distributed_and_scalable::t08_diloco_local_sgd_powersgd_and_distributed_checkpointing as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_diloco_local_sgd_powersgd_and_distributed_checkpointing_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "16_distributed_and_scalable/08_diloco_local_sgd_powersgd_and_distributed_checkpointing.safetensors",
        ),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // DiLoCo
    let (theta_new, v_new, delta_avg) = topic::diloco_outer_step(
        &w["theta_0"],
        &w["theta_workers"],
        &w["v_outer_init"],
        0.7,
        0.9,
    )?;
    common::assert_max_diff(&theta_new, &w["theta_new"], 1e-5)?;
    common::assert_max_diff(&v_new, &w["v_outer_new"], 1e-5)?;
    common::assert_max_diff(&delta_avg, &w["delta_avg"], 1e-5)?;

    // PowerSGD
    let (recon, p_norm, q_new) =
        topic::powersgd_compress_decompress(&w["grad_matrix"], &w["probe_q"])?;
    common::assert_max_diff(&recon, &w["powersgd_recon"], 1e-5)?;
    common::assert_max_diff(&p_norm, &w["powersgd_p"], 1e-5)?;
    common::assert_max_diff(&q_new, &w["powersgd_q"], 1e-5)?;

    // DCP reshard
    let (_, shards_4) = topic::distributed_checkpoint_reshard(&w["shards_2"])?;
    common::assert_max_diff(&shards_4, &w["shards_4"], 1e-5)?;

    Ok(())
}
