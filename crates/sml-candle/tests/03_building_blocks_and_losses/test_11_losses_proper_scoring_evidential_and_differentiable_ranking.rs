//! Integration test for Category 03 Topic 11: Losses Proper Scoring, Evidential, and Differentiable Ranking (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t11_losses_proper_scoring_evidential_and_differentiable_ranking::{
    beta_nll_forward, brier_score_forward, crps_gaussian_forward, edl_mse_forward, listmle_forward,
    soft_ranks_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/11_losses_proper_scoring_evidential_and_differentiable_ranking.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let crps_y = tensors.get("crps_y").expect("crps_y");
    let crps_mu = tensors.get("crps_mu").expect("crps_mu");
    let crps_sigma = tensors.get("crps_sigma").expect("crps_sigma");
    let expected_crps = tensors
        .get("expected_crps")
        .expect("expected_crps")
        .to_vec1::<f32>()?[0];

    let brier_p = tensors.get("brier_p").expect("brier_p");
    let brier_y = tensors.get("brier_y").expect("brier_y");
    let expected_brier = tensors
        .get("expected_brier")
        .expect("expected_brier")
        .to_vec1::<f32>()?[0];

    let beta_nll_y = tensors.get("beta_nll_y").expect("beta_nll_y");
    let beta_nll_mu = tensors.get("beta_nll_mu").expect("beta_nll_mu");
    let beta_nll_var = tensors.get("beta_nll_var").expect("beta_nll_var");
    let expected_beta_nll = tensors
        .get("expected_beta_nll")
        .expect("expected_beta_nll")
        .to_vec1::<f32>()?[0];

    let edl_alpha = tensors.get("edl_alpha").expect("edl_alpha");
    let edl_y = tensors.get("edl_y").expect("edl_y");
    let expected_edl = tensors
        .get("expected_edl")
        .expect("expected_edl")
        .to_vec1::<f32>()?[0];

    let listmle_scores = tensors.get("listmle_scores").expect("listmle_scores");
    let listmle_perm = tensors.get("listmle_perm").expect("listmle_perm");
    let expected_listmle = tensors
        .get("expected_listmle")
        .expect("expected_listmle")
        .to_vec1::<f32>()?[0];

    let soft_scores = tensors.get("soft_scores").expect("soft_scores");
    let expected_soft_ranks = tensors
        .get("expected_soft_ranks")
        .expect("expected_soft_ranks");

    // 1. Linear Anchor Parity
    let anchor = LinearAnchor::new(weight.clone(), bias.clone());
    let linear_pred = anchor.forward(test_in)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. CRPS Gaussian Parity
    let crps_pred = crps_gaussian_forward(crps_y, crps_mu, crps_sigma, 1e-6)?;
    let crps_val = crps_pred.to_vec1::<f32>()?[0];
    let diff_crps = (crps_val - expected_crps).abs();
    assert!(
        diff_crps < 1e-5,
        "CRPS Gaussian parity error {:.2e} exceeds 1e-5",
        diff_crps
    );

    // 3. Brier Score Parity
    let brier_pred = brier_score_forward(brier_p, brier_y)?;
    let brier_val = brier_pred.to_vec1::<f32>()?[0];
    let diff_brier = (brier_val - expected_brier).abs();
    assert!(
        diff_brier < 1e-5,
        "Brier score parity error {:.2e} exceeds 1e-5",
        diff_brier
    );

    // 4. Beta-NLL Parity
    let beta_nll_pred = beta_nll_forward(beta_nll_y, beta_nll_mu, beta_nll_var, 0.5, 1e-6)?;
    let beta_nll_val = beta_nll_pred.to_vec1::<f32>()?[0];
    let diff_beta_nll = (beta_nll_val - expected_beta_nll).abs();
    assert!(
        diff_beta_nll < 1e-5,
        "Beta-NLL parity error {:.2e} exceeds 1e-5",
        diff_beta_nll
    );

    // 5. EDL MSE Parity
    let edl_pred = edl_mse_forward(edl_alpha, edl_y)?;
    let edl_val = edl_pred.to_vec1::<f32>()?[0];
    let diff_edl = (edl_val - expected_edl).abs();
    assert!(
        diff_edl < 1e-5,
        "EDL MSE parity error {:.2e} exceeds 1e-5",
        diff_edl
    );

    // 6. ListMLE Parity
    let listmle_pred = listmle_forward(listmle_scores, listmle_perm)?;
    let listmle_val = listmle_pred.to_vec1::<f32>()?[0];
    let diff_listmle = (listmle_val - expected_listmle).abs();
    assert!(
        diff_listmle < 1e-5,
        "ListMLE parity error {:.2e} exceeds 1e-5",
        diff_listmle
    );

    // 7. Soft Ranks Parity
    let soft_pred = soft_ranks_forward(soft_scores, 0.5)?;
    let diff_soft = soft_pred
        .sub(expected_soft_ranks)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_soft < 1e-5,
        "Soft ranks parity error {:.2e} exceeds 1e-5",
        diff_soft
    );

    Ok(())
}
