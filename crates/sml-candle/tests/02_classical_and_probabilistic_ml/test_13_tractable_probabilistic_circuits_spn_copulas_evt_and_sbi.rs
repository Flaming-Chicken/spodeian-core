use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi as spn;
use std::path::PathBuf;

#[test]
fn test_13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/13_tractable_probabilistic_circuits_spn_copulas_evt_and_sbi.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // 2. Parity Check: SPN Full and Marginal Evaluation
    let leaves_p = tensors.get("leaves_p").expect("Missing leaves_p");
    let prod_weights = tensors.get("prod_weights").expect("Missing prod_weights");
    let ev_full = tensors.get("ev_full").expect("Missing ev_full");
    let ev_marg = tensors.get("ev_marg").expect("Missing ev_marg");
    let golden_spn_full = tensors.get("spn_prob_full").expect("Missing spn_prob_full");
    let golden_spn_marg = tensors.get("spn_prob_marg").expect("Missing spn_prob_marg");

    let spn_full = spn::evaluate_spn(leaves_p, ev_full, prod_weights)?;
    let spn_marg = spn::evaluate_spn(leaves_p, ev_marg, prod_weights)?;

    let diff_full = (&spn_full - golden_spn_full)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_marg = (&spn_marg - golden_spn_marg)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_full < 1e-4, "SPN full diff {diff_full:.2e} >= 1e-4");
    assert!(diff_marg < 1e-4, "SPN marg diff {diff_marg:.2e} >= 1e-4");

    // 3. Parity Check: Clayton Copula CDF
    let u_grid = tensors.get("u_grid").expect("Missing u_grid");
    let v_grid = tensors.get("v_grid").expect("Missing v_grid");
    let golden_clayton = tensors.get("clayton_cdf").expect("Missing clayton_cdf");

    let clayton_res = spn::clayton_copula_cdf(u_grid, v_grid, 2.0)?;
    let diff_copula = (&clayton_res - golden_clayton)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_copula < 1e-4,
        "Clayton copula diff {diff_copula:.2e} >= 1e-4"
    );

    // 4. Parity Check: GPD EVT Tail CDF
    let y_excesses = tensors.get("y_excesses").expect("Missing y_excesses");
    let golden_gpd = tensors.get("gpd_cdf").expect("Missing gpd_cdf");

    let gpd_res = spn::gpd_tail_cdf(y_excesses, 0.25, 1.5)?;
    let diff_gpd = (&gpd_res - golden_gpd)?.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(diff_gpd < 1e-4, "GPD diff {diff_gpd:.2e} >= 1e-4");

    // 5. Parity Check: SBI Neural Ratio
    let x_sbi = tensors.get("x_sbi").expect("Missing x_sbi");
    let theta_sbi = tensors.get("theta_sbi").expect("Missing theta_sbi");
    let w_sbi = tensors.get("w_sbi").expect("Missing w_sbi");
    let b_sbi = tensors.get("b_sbi").expect("Missing b_sbi");
    let golden_ratio = tensors.get("sbi_ratio").expect("Missing sbi_ratio");
    let golden_logit = tensors.get("sbi_logit").expect("Missing sbi_logit");

    let (ratio_res, logit_res) = spn::sbi_neural_ratio(x_sbi, theta_sbi, w_sbi, b_sbi)?;
    let diff_ratio = (&ratio_res - golden_ratio)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_logit = (&logit_res - golden_logit)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_ratio < 1e-4, "SBI ratio diff {diff_ratio:.2e} >= 1e-4");
    assert!(diff_logit < 1e-4, "SBI logit diff {diff_logit:.2e} >= 1e-4");

    Ok(())
}
