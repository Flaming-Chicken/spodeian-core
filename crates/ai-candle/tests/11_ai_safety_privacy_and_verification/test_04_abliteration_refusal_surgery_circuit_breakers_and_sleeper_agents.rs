use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/04_abliteration_refusal_surgery_circuit_breakers_and_sleeper_agents.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Refusal direction
    let r_hat = topic::extract_refusal_direction(&w["act_harmful"], &w["act_harmless"])?;
    common::assert_max_diff(&r_hat, &w["refusal_dir_ref"], 1e-5)?;

    // 3. Abliterate weights
    let w_abl = topic::abliterate_weights(&w["w_mlp"], &r_hat)?;
    common::assert_max_diff(&w_abl, &w["w_ablated_ref"], 1e-5)?;

    // 4. CAA steering
    let h_steered = topic::contrastive_activation_addition(&w["h_query"], &r_hat, 2.0)?;
    common::assert_max_diff(&h_steered, &w["h_steered_ref"], 1e-5)?;

    // 5. Circuit Breakers
    let cb_val = topic::circuit_breaker_loss(
        &w["h_harmful"],
        &w["h_target_ortho"],
        &w["h_benign"],
        &w["h_ref_benign"],
        1.5,
    )?;
    common::assert_max_diff(&cb_val, &w["cb_loss_ref"], 1e-5)?;

    // 6. Sleeper probe
    let scores = topic::sleeper_agent_probe(&w["h_query"], &w["probe_w"], &w["probe_b"])?;
    common::assert_max_diff(&scores, &w["probe_scores_ref"], 1e-5)?;

    Ok(())
}
