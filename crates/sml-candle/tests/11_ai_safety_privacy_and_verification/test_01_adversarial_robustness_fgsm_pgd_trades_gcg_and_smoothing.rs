use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/01_adversarial_robustness_fgsm_pgd_trades_gcg_and_smoothing.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. FGSM perturbation
    let fgsm = topic::fgsm_perturbation(&w["adv_x"], &w["adv_grad"], 0.1, 0.0, 1.0)?;
    common::assert_max_diff(&fgsm, &w["fgsm_adv_ref"], 1e-5)?;

    // 3. PGD step
    let pgd = topic::pgd_step(&fgsm, &w["adv_x"], &w["adv_grad"], 0.04, 0.1, 0.0, 1.0)?;
    common::assert_max_diff(&pgd, &w["pgd_step_ref"], 1e-5)?;

    // 4. TRADES loss
    let t_loss = topic::trades_loss(&w["clean_logits"], &w["adv_logits"], &w["labels"], 1.0)?;
    common::assert_max_diff(&t_loss, &w["trades_loss_ref"], 1e-5)?;

    // 5. Randomized smoothing radius
    let radius = topic::randomized_smoothing_radius(&w["p_top"], &w["p_runnerup"], 0.25)?;
    common::assert_max_diff(&radius, &w["smoothing_radius_ref"], 1e-5)?;

    Ok(())
}
