use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t34_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_34_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/06_graphs_geometric_and_inr/02_graph_neural_networks_gcn_gat_gin_graphormer_and_sdrf_rewiring.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // GCN Normalized adjacency
    let norm_adj = topic::gcn_normalize_adjacency(&w["adj"])?;
    common::assert_max_diff(&norm_adj, &w["norm_adj"], 1e-5)?;

    // GCN layer
    let h_gcn = topic::gcn_layer(&w["h"], &norm_adj, &w["w_gcn"], &w["b_gcn"])?;
    common::assert_max_diff(&h_gcn, &w["h_gcn"], 1e-5)?;

    // GIN layer
    let h_gin = topic::gin_layer(
        &w["h"],
        &w["adj"],
        0.05,
        &w["w1_gin"],
        &w["b1_gin"],
        &w["w2_gin"],
        &w["b2_gin"],
    )?;
    common::assert_max_diff(&h_gin, &w["h_gin"], 1e-5)?;

    // Graphormer attention
    let h_graphormer =
        topic::graphormer_attention(&w["h"], &w["w_q"], &w["w_k"], &w["w_v"], &w["spatial_bias"])?;
    common::assert_max_diff(&h_graphormer, &w["h_graphormer"], 1e-5)?;

    // SDRF Ricci curvature
    let ricci_curv = topic::sdrf_ricci_curvature(&w["adj"])?;
    common::assert_max_diff(&ricci_curv, &w["ricci_curv"], 1e-5)?;

    Ok(())
}
