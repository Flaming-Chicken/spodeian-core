use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t03_vision_optical_flow_depth_and_super_resolution as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_vision_optical_flow_depth_and_super_resolution_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/03_vision_optical_flow_depth_and_super_resolution.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. PixelShuffle
    let ps = topic::pixel_shuffle(&w["sr_input"], 2)?;
    common::assert_max_diff(&ps, &w["sr_output"], 1e-5)?;

    // 2. Scale-Invariant Depth Loss
    let depth_loss =
        topic::scale_invariant_depth_loss(&w["pred_depth"], &w["target_depth"], 0.5, 1e-4)?;
    common::assert_max_diff(&depth_loss, &w["si_depth_ref"], 1e-5)?;

    // 3. Charbonnier Flow Loss
    let flow_loss = topic::charbonnier_loss(&w["pred_flow"], &w["target_flow"], 1e-3)?;
    common::assert_max_diff(&flow_loss, &w["charb_flow_ref"], 1e-5)?;

    // 4. RAFT Correlation Matrix
    let corr = topic::raft_correlation(&w["fmap1"], &w["fmap2"])?;
    common::assert_max_diff(&corr, &w["corr_ref"], 1e-5)?;

    Ok(())
}
