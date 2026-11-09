use candle_core::{safetensors::load, Device, Result};

use ai_candle::c16_distributed_and_scalable::t01_collectives_and_ddp_gradient_bucketing_mechanics as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_collectives_and_ddp_gradient_bucketing_mechanics_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("16_distributed_and_scalable/01_collectives_and_ddp_gradient_bucketing_mechanics.safetensors"), &device)?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Ring AllReduce
    let ar_sum = topic::ring_allreduce_simulate(&w["rank_inputs"])?;
    common::assert_max_diff(&ar_sum, &w["allreduce_sum"], 1e-5)?;

    // 2. DDP Bucketed AllReduce
    let (ddp_grads, buckets) = topic::ddp_bucketed_allreduce(&w["rank_grads"], 8)?;
    common::assert_max_diff(&ddp_grads, &w["ddp_grads_out"], 1e-5)?;
    common::assert_max_diff(&buckets[0], &w["bucket_1_reduced"], 1e-5)?;
    common::assert_max_diff(&buckets[1], &w["bucket_2_reduced"], 1e-5)?;

    Ok(())
}
