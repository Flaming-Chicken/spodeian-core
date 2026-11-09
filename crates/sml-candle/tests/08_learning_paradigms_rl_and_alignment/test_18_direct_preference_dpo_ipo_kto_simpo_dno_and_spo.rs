use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t18_direct_preference_dpo_ipo_kto_simpo_dno_and_spo as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_18_direct_preference_dpo_ipo_kto_simpo_dno_and_spo_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/03_direct_preference_dpo_ipo_kto_simpo_dno_and_spo.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. DPO Loss
    let dpo_l = topic::dpo_loss(
        &w["pi_logps_w"],
        &w["pi_logps_l"],
        &w["ref_logps_w"],
        &w["ref_logps_l"],
        0.1,
    )?;
    common::assert_max_diff(&dpo_l, &w["dpo_loss_ref"], 1e-5)?;

    // 3. IPO Loss
    let ipo_l = topic::ipo_loss(
        &w["pi_logps_w"],
        &w["pi_logps_l"],
        &w["ref_logps_w"],
        &w["ref_logps_l"],
        0.1,
        0.5,
    )?;
    common::assert_max_diff(&ipo_l, &w["ipo_loss_ref"], 1e-5)?;

    // 4. SimPO Loss
    let simpo_l = topic::simpo_loss(
        &w["pi_logps_w"],
        &w["pi_logps_l"],
        &w["len_w"],
        &w["len_l"],
        2.0,
        0.5,
    )?;
    common::assert_max_diff(&simpo_l, &w["simpo_loss_ref"], 1e-5)?;

    // 5. KTO Loss
    let kto_l = topic::kto_loss(
        &w["pi_logps_unpaired"],
        &w["ref_logps_unpaired"],
        &w["desirable_flags"],
        0.1,
        1.0,
        1.33,
    )?;
    common::assert_max_diff(&kto_l, &w["kto_loss_ref"], 1e-5)?;

    Ok(())
}
