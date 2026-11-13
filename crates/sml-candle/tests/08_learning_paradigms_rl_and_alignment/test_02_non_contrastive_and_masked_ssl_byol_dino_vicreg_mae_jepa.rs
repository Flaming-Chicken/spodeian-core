use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/02_non_contrastive_and_masked_ssl_byol_dino_vicreg_mae_jepa.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. BYOL loss
    let byol_loss = topic::byol_prediction_loss(&w["byol_p"], &w["byol_z"])?;
    common::assert_max_diff(&byol_loss, &w["byol_loss_ref"], 1e-5)?;

    // 3. VICReg loss
    let vicreg_loss =
        topic::vicreg_loss(&w["vicreg_za"], &w["vicreg_zb"], 1.0, 1.0, 0.04, 1.0, 1e-4)?;
    common::assert_max_diff(&vicreg_loss, &w["vicreg_loss_ref"], 1e-5)?;

    // 4. DINO loss
    let dino_loss = topic::dino_loss(&w["s_logits"], &w["t_logits"], &w["t_center"], 0.1, 0.05)?;
    common::assert_max_diff(&dino_loss, &w["dino_loss_ref"], 1e-5)?;

    // 5. KoLeo loss
    let koleo_loss = topic::koleo_loss(&w["koleo_z"], 1e-8)?;
    common::assert_max_diff(&koleo_loss, &w["koleo_loss_ref"], 1e-5)?;

    // 6. MAE patch gather
    let mae_gathered = topic::mae_patch_gather(&w["mae_patches"], &w["mae_indices"])?;
    common::assert_max_diff(&mae_gathered, &w["mae_gathered_ref"], 1e-5)?;

    // 7. JEPA feature loss
    let jepa_loss = topic::jepa_feature_loss(&w["jepa_pred"], &w["jepa_target"])?;
    common::assert_max_diff(&jepa_loss, &w["jepa_loss_ref"], 1e-5)?;

    Ok(())
}
