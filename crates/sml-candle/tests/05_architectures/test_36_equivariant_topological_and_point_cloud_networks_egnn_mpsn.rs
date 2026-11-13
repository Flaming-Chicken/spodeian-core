use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t36_equivariant_topological_and_point_cloud_networks_egnn_mpsn as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_36_equivariant_topological_and_point_cloud_networks_egnn_mpsn_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // EGNN layer
    let (x_next, h_next) = topic::egnn_layer(
        &w["x"], &w["h"], &w["adj"], &w["w_m"], &w["b_m"], &w["w_x"], &w["b_x"], &w["w_h"],
        &w["b_h"],
    )?;
    common::assert_max_diff(&x_next, &w["x_next"], 1e-5)?;
    common::assert_max_diff(&h_next, &w["h_next"], 1e-5)?;

    // PointNet global pool
    let p_global = topic::pointnet_global_pool(&w["x"], &w["w_p"], &w["b_p"])?;
    common::assert_max_diff(&p_global, &w["p_global"], 1e-5)?;

    Ok(())
}
