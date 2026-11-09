use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t37_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_37_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/06_graphs_geometric_and_inr/05_implicit_neural_representations_siren_instant_ngp_nerf_and_3dgs.safetensors"),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // SIREN layer
    let h_siren = topic::siren_layer(&w["x_siren"], &w["w_siren"], &w["b_siren"], 30.0)?;
    common::assert_max_diff(&h_siren, &w["h_siren"], 1e-5)?;

    // NeRF volume rendering
    let (rendered_color, accum_opacity) =
        topic::volume_rendering_quadrature(&w["densities"], &w["colors"], &w["deltas"])?;
    common::assert_max_diff(&rendered_color, &w["rendered_color"], 1e-5)?;
    common::assert_max_diff(&accum_opacity, &w["accum_opacity"], 1e-5)?;

    // 3DGS Gaussian alpha evaluation
    let splat_alpha = topic::gaussian_splat_2d_eval(&w["x_pixels"], &w["mu"], &w["cov_inv"])?;
    common::assert_max_diff(&splat_alpha, &w["splat_alpha"], 1e-5)?;

    Ok(())
}
