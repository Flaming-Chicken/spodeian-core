use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t39_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_39_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/01_multimodal_fusion_film_perceiver_resampler_and_vlm_connectors.safetensors",
        ),
        &device,
    )?;

    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let h_film = topic::film_modulation(&w["x_feat"], &w["gamma"], &w["beta"])?;
    common::assert_max_diff(&h_film, &w["h_film"], 1e-5)?;

    let h_resampled = topic::perceiver_resampler(
        &w["latent_queries"],
        &w["x_feat"],
        &w["w_q"],
        &w["w_k"],
        &w["w_v"],
    )?;
    common::assert_max_diff(&h_resampled, &w["h_resampled"], 1e-5)?;

    let h_vlm =
        topic::vlm_spatial_pool_connector(&w["patches"], &w["w1"], &w["b1"], &w["w2"], &w["b2"])?;
    common::assert_max_diff(&h_vlm, &w["h_vlm"], 1e-5)?;

    Ok(())
}
