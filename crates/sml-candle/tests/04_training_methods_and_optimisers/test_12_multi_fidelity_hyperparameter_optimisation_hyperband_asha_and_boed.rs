use candle_core::{Device, Result, Tensor, safetensors::load};

use sml_candle::c04_training_methods_and_optimisers::t12_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed as opt;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation/05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed.safetensors",
    );
    let weights = load(&weights_path, &device)?;

    let sha_losses = weights.get("sha_losses").unwrap();
    let golden_sha_idx = weights.get("golden_sha_idx").unwrap();
    let golden_sha_losses = weights.get("golden_sha_losses").unwrap();
    let golden_hb_schedule = weights.get("golden_hb_schedule").unwrap();

    let x_query = weights.get("x_query").unwrap();
    let good_samples = weights.get("good_samples").unwrap();
    let bad_samples = weights.get("bad_samples").unwrap();
    let golden_tpe_log_ratio = weights.get("golden_tpe_log_ratio").unwrap();

    let rung_losses = weights.get("rung_losses").unwrap();
    let candidate_losses = weights.get("candidate_losses").unwrap();
    let golden_asha_promoted = weights.get("golden_asha_promoted").unwrap();

    // 1. SHA
    let (idx, losses) = opt::compute_sha_top_k(sha_losses, 3)?;
    let diff_idx = idx
        .sub(golden_sha_idx)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_l = losses
        .sub(golden_sha_losses)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_idx < 1e-5, "SHA idx diff: {}", diff_idx);
    assert!(diff_l < 1e-5, "SHA loss diff: {}", diff_l);

    // 2. Hyperband
    let schedule = opt::compute_hyperband_schedule(81, 3, &device)?;
    let diff_hb = schedule
        .sub(golden_hb_schedule)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_hb < 1e-5, "Hyperband diff: {}", diff_hb);

    // 3. TPE
    let tpe_ratio = opt::compute_tpe_log_ratio(x_query, good_samples, bad_samples, 0.5)?;
    let diff_tpe = tpe_ratio
        .sub(golden_tpe_log_ratio)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_tpe < 1e-5, "TPE diff: {}", diff_tpe);

    // 4. ASHA
    let cand_vec = candidate_losses.to_vec1::<f32>()?;
    let p0 = opt::check_asha_promotion(rung_losses, cand_vec[0] as f64, 3)?;
    let p1 = opt::check_asha_promotion(rung_losses, cand_vec[1] as f64, 3)?;
    let asha_promoted = Tensor::from_vec(vec![p0 as f32, p1 as f32], 2, &device)?;
    let diff_asha = asha_promoted
        .sub(golden_asha_promoted)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_asha < 1e-5, "ASHA diff: {}", diff_asha);

    // 5. Linear Anchor
    let anchor_w = weights.get("anchor_w").unwrap();
    let anchor_b = weights.get("anchor_b").unwrap();
    let anchor_x = weights.get("anchor_x").unwrap();
    let anchor = opt::LinearAnchor::new(anchor_w.clone(), anchor_b.clone());
    let anchor_out = anchor.forward(anchor_x)?;
    let expected_anchor = weights.get("golden_anchor_out").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "anchor diff: {}", diff_anchor);

    // Unified module verification
    let module = opt::MultiFidelityUnifiedModule::load_from_safetensors(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;
    assert!(errs["sha_err"] < 1e-5);
    assert!(errs["hyperband_err"] < 1e-5);
    assert!(errs["tpe_err"] < 1e-5);
    assert!(errs["asha_err"] < 1e-5);
    assert!(errs["anchor_err"] < 1e-5);

    Ok(())
}
