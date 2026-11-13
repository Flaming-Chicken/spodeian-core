use candle_core::{Device, Result, safetensors::load};

use sml_candle::c11_ai_safety_privacy_and_verification::t06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "11_ai_safety_privacy_and_verification/06_backdoor_trojan_and_poisoning_defense_neural_cleanse_and_spectral.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Spectral signatures
    let spec_scores = topic::spectral_signature_scores(&w["representations"])?;
    common::assert_max_diff(&spec_scores, &w["spectral_scores_ref"], 1e-5)?;

    // 3. Neural Cleanse MAD anomaly
    let mad_anomaly = topic::neural_cleanse_anomaly_index(&w["l1_norms"])?;
    common::assert_max_diff(&mad_anomaly, &w["mad_anomaly_ref"], 1e-5)?;

    // 4. STRIP entropy
    let h_benign = topic::strip_entropy(&w["logits_benign"])?;
    common::assert_max_diff(&h_benign, &w["entropy_benign_ref"], 1e-5)?;

    let h_trojan = topic::strip_entropy(&w["logits_trojan"])?;
    common::assert_max_diff(&h_trojan, &w["entropy_trojan_ref"], 1e-5)?;

    // 5. Trigger apply
    let x_trojan = topic::trojan_trigger_apply(&w["x_input"], &w["mask"], &w["pattern"])?;
    common::assert_max_diff(&x_trojan, &w["x_trojan_ref"], 1e-5)?;

    Ok(())
}
