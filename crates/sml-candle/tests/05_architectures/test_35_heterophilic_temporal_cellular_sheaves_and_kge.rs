use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t35_heterophilic_temporal_cellular_sheaves_and_kge as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_35_heterophilic_temporal_cellular_sheaves_and_kge_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/06_graphs_geometric_and_inr/03_heterophilic_temporal_cellular_sheaves_and_kge.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Heterophilic filter
    let h_hetero = topic::heterophilic_filter(
        &w["h"],
        &w["norm_adj"],
        &w["w_self"],
        &w["w_low"],
        &w["w_high"],
    )?;
    common::assert_max_diff(&h_hetero, &w["h_hetero"], 1e-5)?;

    // Sheaf Laplacian
    let sheaf_laplacian = topic::cellular_sheaf_laplacian_1d(&w["adj"], &w["edge_signs"])?;
    common::assert_max_diff(&sheaf_laplacian, &w["sheaf_laplacian"], 1e-5)?;

    // Sheaf diffusion step
    let h_sheaf_diff = topic::sheaf_diffusion_step(&w["h"], &sheaf_laplacian, 0.1)?;
    common::assert_max_diff(&h_sheaf_diff, &w["h_sheaf_diff"], 1e-5)?;

    // RotatE KGE score
    let rotate_score =
        topic::rotate_kge_score(&w["h_re"], &w["h_im"], &w["theta"], &w["t_re"], &w["t_im"])?;
    common::assert_max_diff(&rotate_score, &w["rotate_score"], 1e-5)?;

    Ok(())
}
