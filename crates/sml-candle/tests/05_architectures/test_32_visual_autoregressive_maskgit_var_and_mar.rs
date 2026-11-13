use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t32_visual_autoregressive_maskgit_var_and_mar as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_32_visual_autoregressive_maskgit_var_and_mar_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/09_visual_autoregressive_maskgit_var_and_mar.safetensors",
        ),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // MaskGIT cosine schedule
    let gamma = topic::maskgit_cosine_schedule(&w["r"])?;
    common::assert_max_diff(&gamma, &w["gamma"], 1e-5)?;

    // MaskGIT apply mask
    let masked_tokens = topic::maskgit_apply_mask(&w["x_tokens"], &w["mask"], &w["mask_token"])?;
    common::assert_max_diff(&masked_tokens, &w["masked_tokens"], 1e-5)?;

    // VAR coarse-to-fine upsample
    let fine_map =
        topic::var_coarse_to_fine_upsample(&w["coarse_map"], 4, &w["w_scale"], &w["b_scale"])?;
    common::assert_max_diff(&fine_map, &w["fine_map"], 1e-5)?;

    // MAR diffusion loss
    let mar_loss = topic::mar_masked_diffusion_loss(&w["v_pred"], &w["v_target"], &w["mask"])?;
    common::assert_max_diff(&mar_loss, &w["mar_loss"], 1e-5)?;

    Ok(())
}
