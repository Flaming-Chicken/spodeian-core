use candle_core::{Device, Result, safetensors::load};

use sml_candle::c11_ai_safety_privacy_and_verification::t09_ood_detection_abstention_conformal_and_algorithmic_fairness as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_ood_detection_abstention_conformal_and_algorithmic_fairness_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "11_ai_safety_privacy_and_verification/09_ood_detection_abstention_conformal_and_algorithmic_fairness.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Energy OOD
    let energy = topic::energy_ood_score(&w["logits"], 1.0)?;
    common::assert_max_diff(&energy, &w["energy_ref"], 1e-5)?;

    // 3. Conformal quantile
    let q_hat = topic::split_conformal_quantile(&w["cal_scores"], 0.1)?;
    common::assert_max_diff(&q_hat, &w["q_hat_ref"], 1e-5)?;

    // 4. SelectiveNet loss
    let sel_loss = topic::selective_net_loss(&w["ce_losses"], &w["selection_probs"], 0.7, 5.0)?;
    common::assert_max_diff(&sel_loss, &w["selective_loss_ref"], 1e-5)?;

    // 5. Equalized Odds discrepancy
    let eo_disc =
        topic::equalized_odds_discrepancy(&w["y_pred"], &w["y_true"], &w["sensitive_attr"])?;
    common::assert_max_diff(&eo_disc, &w["eo_discrepancy_ref"], 1e-5)?;

    Ok(())
}
