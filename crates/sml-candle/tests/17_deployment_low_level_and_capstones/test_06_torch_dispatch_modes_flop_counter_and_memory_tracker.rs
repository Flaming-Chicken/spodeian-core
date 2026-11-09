use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t06_torch_dispatch_modes_flop_counter_and_memory_tracker as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_torch_dispatch_modes_flop_counter_and_memory_tracker_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/06_torch_dispatch_modes_flop_counter_and_memory_tracker.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let (y, z1) = topic::mlp_forward(&w["x"], &w["w1"], &w["b1"], &w["w2"], &w["b2"])?;
    common::assert_max_diff(&y, &w["y"], 1e-5)?;
    common::assert_max_diff(&z1, &w["z1"], 1e-5)?;

    Ok(())
}
