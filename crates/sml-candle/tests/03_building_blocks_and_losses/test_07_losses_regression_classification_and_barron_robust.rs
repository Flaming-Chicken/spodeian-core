//! Integration test for Category 03 Topic 07: Losses Regression, Classification, and Barron Robust (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t07_losses_regression_classification_and_barron_robust::{
    asymmetric_loss_forward, barron_robust_loss_forward, focal_loss_forward, huber_loss_forward,
    polyloss_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/07_losses_regression_classification_and_barron_robust.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let r_barron = tensors.get("r_barron").expect("r_barron");
    let alpha_b = tensors
        .get("alpha_barron")
        .expect("alpha_barron")
        .to_vec1::<f32>()?[0] as f64;
    let c_b = tensors
        .get("c_barron")
        .expect("c_barron")
        .to_vec1::<f32>()?[0] as f64;
    let loss_barron_gold = tensors
        .get("loss_barron")
        .expect("loss_barron")
        .to_vec1::<f32>()?[0];

    let r_huber = tensors.get("r_huber").expect("r_huber");
    let delta_h = tensors
        .get("delta_huber")
        .expect("delta_huber")
        .to_vec1::<f32>()?[0] as f64;
    let loss_huber_gold = tensors
        .get("loss_huber")
        .expect("loss_huber")
        .to_vec1::<f32>()?[0];

    let logits_clf = tensors.get("logits_clf").expect("logits_clf");
    let targets_clf = tensors.get("targets_clf").expect("targets_clf");
    let gamma_f = tensors
        .get("gamma_focal")
        .expect("gamma_focal")
        .to_vec1::<f32>()?[0] as f64;
    let loss_focal_gold = tensors
        .get("loss_focal")
        .expect("loss_focal")
        .to_vec1::<f32>()?[0];

    let eps_p = tensors
        .get("eps_poly")
        .expect("eps_poly")
        .to_vec1::<f32>()?[0] as f64;
    let loss_poly_gold = tensors
        .get("loss_poly")
        .expect("loss_poly")
        .to_vec1::<f32>()?[0];

    let logits_asl = tensors.get("logits_asl").expect("logits_asl");
    let targets_asl = tensors.get("targets_asl").expect("targets_asl");
    let gamma_pos = tensors
        .get("gamma_pos")
        .expect("gamma_pos")
        .to_vec1::<f32>()?[0] as f64;
    let gamma_neg = tensors
        .get("gamma_neg")
        .expect("gamma_neg")
        .to_vec1::<f32>()?[0] as f64;
    let margin_asl = tensors
        .get("margin_asl")
        .expect("margin_asl")
        .to_vec1::<f32>()?[0] as f64;
    let loss_asl_gold = tensors
        .get("loss_asl")
        .expect("loss_asl")
        .to_vec1::<f32>()?[0];

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

    // 2. Barron Robust Loss Parity
    let loss_b_calc = barron_robust_loss_forward(r_barron, alpha_b, c_b)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_barron = (loss_b_calc - loss_barron_gold).abs();
    assert!(
        diff_barron < 1e-5,
        "Barron robust loss parity error {:.2e} exceeds 1e-5",
        diff_barron
    );

    // 3. Huber Loss Parity
    let loss_h_calc = huber_loss_forward(r_huber, delta_h)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_huber = (loss_h_calc - loss_huber_gold).abs();
    assert!(
        diff_huber < 1e-5,
        "Huber loss parity error {:.2e} exceeds 1e-5",
        diff_huber
    );

    // 4. Focal Loss Parity
    let loss_f_calc = focal_loss_forward(logits_clf, targets_clf, gamma_f)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_focal = (loss_f_calc - loss_focal_gold).abs();
    assert!(
        diff_focal < 1e-5,
        "Focal loss parity error {:.2e} exceeds 1e-5",
        diff_focal
    );

    // 5. PolyLoss Parity
    let loss_p_calc = polyloss_forward(logits_clf, targets_clf, eps_p)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_poly = (loss_p_calc - loss_poly_gold).abs();
    assert!(
        diff_poly < 1e-5,
        "PolyLoss parity error {:.2e} exceeds 1e-5",
        diff_poly
    );

    // 6. Asymmetric Loss Parity
    let loss_asl_calc = asymmetric_loss_forward(
        logits_asl,
        targets_asl,
        gamma_pos,
        gamma_neg,
        margin_asl,
        1e-8,
    )?
    .flatten_all()?
    .to_vec1::<f32>()?[0];
    let diff_asl = (loss_asl_calc - loss_asl_gold).abs();
    assert!(
        diff_asl < 1e-5,
        "Asymmetric loss parity error {:.2e} exceeds 1e-5",
        diff_asl
    );

    Ok(())
}
