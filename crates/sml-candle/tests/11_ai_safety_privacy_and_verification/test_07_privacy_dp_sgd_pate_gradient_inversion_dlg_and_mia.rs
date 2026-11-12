use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/07_privacy_dp_sgd_pate_gradient_inversion_dlg_and_mia.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. DP-SGD clip & noise
    let dp_grad = topic::dp_sgd_clip_and_noise(&w["per_sample_grads"], 1.0, &w["noise_vec"], 0.5)?;
    common::assert_max_diff(&dp_grad, &w["dp_grad_ref"], 1e-5)?;

    // 3. PATE noisy aggregation
    let pate_votes = topic::pate_noisy_aggregation(&w["teacher_votes"], &w["laplace_noise"], 0.5)?;
    common::assert_max_diff(&pate_votes, &w["pate_votes_ref"], 1e-5)?;

    // 4. DLG gradient inversion loss
    let dlg_loss = topic::dlg_gradient_inversion_loss(&w["dummy_grad"], &w["target_grad"])?;
    common::assert_max_diff(&dlg_loss, &w["dlg_loss_ref"], 1e-5)?;

    // 5. LiRA membership score
    let lira_scores = topic::lira_membership_score(&w["target_loss"], &w["shadow_out_losses"])?;
    common::assert_max_diff(&lira_scores, &w["lira_scores_ref"], 1e-5)?;

    Ok(())
}
