use candle_core::{safetensors::load, Device, Result};

use ai_candle::c16_distributed_and_scalable::t02_device_mesh_and_dtensor_sharding_placements as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_device_mesh_and_dtensor_sharding_placements_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("16_distributed_and_scalable/02_device_mesh_and_dtensor_sharding_placements.safetensors"), &device)?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Shard & Gather roundtrip
    let x_sharded = topic::dtensor_shard(&w["global_x"], 2, 0)?;
    common::assert_max_diff(&x_sharded, &w["x_sharded_dim0"], 1e-5)?;
    let x_gathered = topic::dtensor_allgather(&x_sharded, 0)?;
    common::assert_max_diff(&x_gathered, &w["x_gathered"], 1e-5)?;

    // 2. Column-parallel matmul
    let col_out = topic::dtensor_column_parallel_matmul(&w["global_x"], &w["w_col_shards"])?;
    common::assert_max_diff(&col_out, &w["col_out_shards"], 1e-5)?;
    let col_gathered = topic::dtensor_allgather(&col_out, 1)?;
    common::assert_max_diff(&col_gathered, &w["col_out_gathered"], 1e-5)?;

    // 3. Row-parallel matmul
    let (row_allred, partials) =
        topic::dtensor_row_parallel_matmul(&w["x_row_shards"], &w["w_row_shards"])?;
    common::assert_max_diff(&partials, &w["partial_shards"], 1e-5)?;
    common::assert_max_diff(&row_allred, &w["row_out_allreduced"], 1e-5)?;

    Ok(())
}
