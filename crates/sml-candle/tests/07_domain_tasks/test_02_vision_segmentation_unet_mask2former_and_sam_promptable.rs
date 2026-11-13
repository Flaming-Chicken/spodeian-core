use candle_core::{Device, Result, safetensors::load};

use ai_candle::c07_domain_tasks::t02_vision_segmentation_unet_mask2former_and_sam_promptable as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_vision_segmentation_unet_mask2former_and_sam_promptable_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/02_vision_segmentation_unet_mask2former_and_sam_promptable.safetensors",
        ),
        &device,
    )?;

    // 0. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Soft Dice Loss
    let dice = topic::dice_loss(&w["pred_mask"], &w["target_mask"], 1e-5)?;
    common::assert_max_diff(&dice, &w["dice_loss_ref"], 1e-5)?;

    // 2. Mask2Former Masked Cross-Attention
    let m2f = topic::mask2former_masked_attention(&w["q"], &w["k"], &w["v"], &w["attn_mask"])?;
    common::assert_max_diff(&m2f, &w["m2f_out"], 1e-5)?;

    // 3. SAM Fourier Prompt Encoder
    let sam = topic::sam_fourier_prompt_embed(&w["points"], &w["gaussian_matrix"])?;
    common::assert_max_diff(&sam, &w["sam_prompt_embed"], 1e-5)?;

    Ok(())
}
