use candle_core::{safetensors::load, Device, Result};

use ai_candle::c09_causal_ai_and_evolutionary_search::t03_domain_generalization_irm_groupdro_vrex_coral_and_dann as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_domain_generalization_irm_groupdro_vrex_coral_and_dann_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("09_causal_ai_and_evolutionary_search/03_domain_generalization_irm_groupdro_vrex_coral_and_dann.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. IRMv1 penalty
    let irm_pen = topic::irm_v1_penalty(&w["irm_logits"], &w["irm_targets"])?;
    common::assert_max_diff(&irm_pen, &w["irm_penalty_ref"], 1e-5)?;

    // 3. V-REx variance penalty
    let vrex = topic::vrex_variance_penalty(&w["env_losses"])?;
    common::assert_max_diff(&vrex, &w["vrex_penalty_ref"], 1e-5)?;

    // 4. GroupDRO update
    let (dro_loss, q_up) = topic::group_dro_update(&w["env_losses"], &w["q_weights"], 0.1)?;
    common::assert_max_diff(&dro_loss, &w["dro_loss_ref"], 1e-5)?;
    common::assert_max_diff(&q_up, &w["q_updated_ref"], 1e-5)?;

    // 5. Deep CORAL covariance loss
    let coral = topic::deep_coral_loss(&w["phi_source"], &w["phi_target"])?;
    common::assert_max_diff(&coral, &w["coral_loss_ref"], 1e-5)?;

    Ok(())
}
