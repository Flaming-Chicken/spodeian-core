use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t03_machine_unlearning_sisa_influence_scrub_npo_and_rmu as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_machine_unlearning_sisa_influence_scrub_npo_and_rmu_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/03_machine_unlearning_sisa_influence_scrub_npo_and_rmu.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. NPO loss
    let npo_val = topic::npo_loss(&w["logits_theta"], &w["logits_ref"], &w["labels"], 0.2)?;
    common::assert_max_diff(&npo_val, &w["npo_ref"], 1e-5)?;

    // 3. RMU loss
    let rmu_val = topic::rmu_loss(
        &w["h_theta_forget"],
        &w["h_target_random"],
        &w["h_theta_retain"],
        &w["h_ref_retain"],
        1.5,
    )?;
    common::assert_max_diff(&rmu_val, &w["rmu_ref"], 1e-5)?;

    // 4. SCRUB loss
    let scrub_val = topic::scrub_loss(
        &w["logits_f_theta"],
        &w["labels_f"],
        &w["logits_r_theta"],
        &w["logits_r_ref"],
        1.2,
        4.0,
    )?;
    common::assert_max_diff(&scrub_val, &w["scrub_ref"], 1e-5)?;

    // 5. SISA ensemble
    let sisa_val = topic::sisa_ensemble_predict(&w["shard_logits"])?;
    common::assert_max_diff(&sisa_val, &w["sisa_ref"], 1e-5)?;

    Ok(())
}
