use candle_core::{safetensors::load, Device, Result};

use ai_candle::c06_smooth_and_physics_informed_ai::t01_lipschitz_networks_bjorck_cayley_and_groupsort as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_lipschitz_networks_bjorck_cayley_and_groupsort_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort.safetensors"), &device)?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let w_ortho = topic::bjorck_orthonormalize(&w["w_raw"], 5)?;
    common::assert_max_diff(&w_ortho, &w["w_ortho"], 1e-5)?;

    let q_cayley = topic::cayley_transform(&w["m_raw"])?;
    common::assert_max_diff(&q_cayley, &w["q_cayley"], 1e-5)?;

    let h_bjorck = w["x_feat"].matmul(&w_ortho)?;
    common::assert_max_diff(&h_bjorck, &w["h_bjorck"], 1e-5)?;

    let y_groupsort = topic::groupsort2(&w["h_bjorck"])?;
    common::assert_max_diff(&y_groupsort, &w["y_groupsort"], 1e-5)?;

    Ok(())
}
