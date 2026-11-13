use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t01_contrastive_ssl_simclr_moco_and_clip_siglip as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_contrastive_ssl_simclr_moco_and_clip_siglip_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/01_contrastive_ssl_simclr_moco_and_clip_siglip.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. SimCLR
    let simclr = topic::simclr_nt_xent_loss(&w["z1"], &w["z2"], 0.1)?;
    common::assert_max_diff(&simclr, &w["simclr_ref"], 1e-5)?;

    // 2. MoCo
    let moco = topic::moco_contrastive_loss(&w["moco_q"], &w["moco_k_pos"], &w["moco_queue"], 0.1)?;
    common::assert_max_diff(&moco, &w["moco_ref"], 1e-5)?;

    // 3. CLIP
    let clip = topic::clip_symmetric_loss(
        &w["img_embeds"],
        &w["txt_embeds"],
        &w["log_temp"],
        &w["clip_bias"],
    )?;
    common::assert_max_diff(&clip, &w["clip_ref"], 1e-5)?;

    // 4. SigLIP
    let siglip = topic::siglip_pairwise_loss(
        &w["img_embeds"],
        &w["txt_embeds"],
        &w["log_temp"],
        &w["clip_bias"],
    )?;
    common::assert_max_diff(&siglip, &w["siglip_ref"], 1e-5)?;

    Ok(())
}
