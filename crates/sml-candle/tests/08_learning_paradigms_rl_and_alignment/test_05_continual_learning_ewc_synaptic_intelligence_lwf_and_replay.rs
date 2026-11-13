use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/05_continual_learning_ewc_synaptic_intelligence_lwf_and_replay.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. EWC
    let penalty = topic::ewc_penalty(&w["theta"], &w["theta_star"], &w["fisher_diag"], 10.0)?;
    common::assert_max_diff(&penalty, &w["ewc_penalty_ref"], 1e-5)?;

    // 3. SI
    let omega = topic::synaptic_intelligence_omega(&w["delta_theta"], &w["path_integral"], 0.1)?;
    common::assert_max_diff(&omega, &w["si_omega_ref"], 1e-5)?;

    // 4. LwF
    let lwf = topic::lwf_distillation_loss(&w["new_logits"], &w["old_logits"], 2.0)?;
    common::assert_max_diff(&lwf, &w["lwf_loss_ref"], 1e-5)?;

    // 5. DER++
    let der = topic::der_dark_replay_loss(
        &w["current_logits"],
        &w["past_logits"],
        &w["labels"],
        0.5,
        0.5,
    )?;
    common::assert_max_diff(&der, &w["der_loss_ref"], 1e-5)?;

    Ok(())
}
