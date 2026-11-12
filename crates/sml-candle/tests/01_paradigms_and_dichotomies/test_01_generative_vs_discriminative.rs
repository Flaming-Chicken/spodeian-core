//! Integration test for Category 01 Topic 01: Generative vs Discriminative Classifiers (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t01_generative_vs_discriminative_classifiers::{
    DiscriminativeClassifier, GaussianDiscriminantAnalysis,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/01_generative_vs_discriminative_classifiers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");

    let w_gda = tensors.get("layers.gda_weight").expect("layers.gda_weight");
    let b_gda = tensors.get("layers.gda_bias").expect("layers.gda_bias");
    let gda_logits_gold = tensors.get("layers.gda_logits").expect("layers.gda_logits");

    // 1. Discriminative Forward Prediction Parity
    let disc_model = DiscriminativeClassifier::new(weight_gold.clone(), bias_gold.clone());
    let disc_pred = disc_model.forward(test_in)?;
    let diff_disc = disc_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_disc < 1e-5,
        "Discriminative prediction error {:.2e} exceeds 1e-5",
        diff_disc
    );

    // 2. Generative GDA Prediction Parity
    let gda_model = GaussianDiscriminantAnalysis::new(w_gda.clone(), b_gda.clone());
    let gda_pred = gda_model.predict_logits(test_in)?;
    let diff_gda = gda_pred
        .sub(gda_logits_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gda < 1e-5,
        "Generative GDA logits error {:.2e} exceeds 1e-5",
        diff_gda
    );

    // 3. Probability Distribution Axiom Invariant Check
    let probs = disc_model.predict_proba(test_in, &device)?;
    let prob_sums = probs.sum(1)?;
    let sum_diff = prob_sums
        .sub(&candle_core::Tensor::ones_like(&prob_sums)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        sum_diff < 1e-5,
        "Probability sum error {:.2e} exceeds 1e-5",
        sum_diff
    );

    Ok(())
}
