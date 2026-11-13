//! Integration test for Category 01 Topic 04: Deterministic vs Stochastic & Mixture Density Networks (Candle Track).

use sml_candle::c01_paradigms_and_dichotomies::t04_deterministic_vs_stochastic_and_mixture_density_networks::{
    DeterministicRegressor, MixtureDensityNetwork, QuantileRegressor,
};
use candle_core::{safetensors::load, Device, Result, Tensor};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/04_deterministic_vs_stochastic_and_mixture_density_networks.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");

    // 1. Canonical Linear Anchor Parity
    let linear_pred = test_in.matmul(weight)?.broadcast_add(bias)?;
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

    // 2. Shared Backbone & Deterministic Regressor Parity
    let w_h = tensors.get("layers.w_h").expect("w_h");
    let b_h = tensors.get("layers.b_h").expect("b_h");
    let w_det = tensors.get("layers.w_det").expect("w_det");
    let b_det = tensors.get("layers.b_det").expect("b_det");
    let y_det_gold = tensors.get("layers.y_det").expect("y_det");

    let det_model =
        DeterministicRegressor::new(w_h.clone(), b_h.clone(), w_det.clone(), b_det.clone());
    let y_det_calc = det_model.forward(test_in)?;
    let diff_det = y_det_calc
        .sub(y_det_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_det < 1e-5,
        "Deterministic regressor parity error {:.2e} exceeds 1e-5",
        diff_det
    );

    // 3. Mixture Density Network Parity (pi, mu, sigma, and log_prob)
    let w_pi = tensors.get("layers.w_pi").expect("w_pi");
    let b_pi = tensors.get("layers.b_pi").expect("b_pi");
    let w_mu = tensors.get("layers.w_mu").expect("w_mu");
    let b_mu = tensors.get("layers.b_mu").expect("b_mu");
    let w_sigma = tensors.get("layers.w_sigma").expect("w_sigma");
    let b_sigma = tensors.get("layers.b_sigma").expect("b_sigma");

    let pi_gold = tensors.get("layers.pi").expect("pi");
    let mu_gold = tensors.get("layers.mu").expect("mu");
    let sigma_gold = tensors.get("layers.sigma").expect("sigma");
    let sample_y = tensors.get("layers.sample_y").expect("sample_y");
    let log_probs_gold = tensors.get("layers.mdn_log_probs").expect("mdn_log_probs");

    let mdn = MixtureDensityNetwork::new(
        w_h.clone(),
        b_h.clone(),
        w_pi.clone(),
        b_pi.clone(),
        w_mu.clone(),
        b_mu.clone(),
        w_sigma.clone(),
        b_sigma.clone(),
    );
    let (pi_calc, mu_calc, sigma_calc) = mdn.forward(test_in)?;

    let diff_pi = pi_calc
        .sub(pi_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pi < 1e-5,
        "MDN pi parity error {:.2e} exceeds 1e-5",
        diff_pi
    );

    let diff_mu = mu_calc
        .sub(mu_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mu < 1e-5,
        "MDN mu parity error {:.2e} exceeds 1e-5",
        diff_mu
    );

    let diff_sigma = sigma_calc
        .sub(sigma_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sigma < 1e-5,
        "MDN sigma parity error {:.2e} exceeds 1e-5",
        diff_sigma
    );

    let log_probs_calc = mdn.log_prob(test_in, sample_y)?;
    let diff_log_probs = log_probs_calc
        .sub(log_probs_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_log_probs < 1e-5,
        "MDN log_prob parity error {:.2e} exceeds 1e-5",
        diff_log_probs
    );

    // 4. Quantile Regressor Parity & Pinball Loss
    let w_quantile = tensors.get("layers.w_quantile").expect("w_quantile");
    let b_quantile = tensors.get("layers.b_quantile").expect("b_quantile");
    let quantiles = tensors.get("layers.quantiles").expect("quantiles");
    let y_quantiles_gold = tensors.get("layers.y_quantiles").expect("y_quantiles");
    let pinball_gold = tensors.get("layers.pinball_loss").expect("pinball_loss");

    let q_model = QuantileRegressor::new(
        w_h.clone(),
        b_h.clone(),
        w_quantile.clone(),
        b_quantile.clone(),
    );
    let y_quantiles_calc = q_model.forward(test_in)?;
    let diff_q = y_quantiles_calc
        .sub(y_quantiles_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_q < 1e-5,
        "Quantile forward parity error {:.2e} exceeds 1e-5",
        diff_q
    );

    let pinball_calc = QuantileRegressor::pinball_loss(&y_quantiles_calc, sample_y, quantiles)?;
    let diff_pinball = pinball_calc
        .sub(pinball_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pinball < 1e-5,
        "Pinball loss parity error {:.2e} exceeds 1e-5",
        diff_pinball
    );

    // 5. Simplex Invariant for mixing coefficients: sum(pi, -1) == 1
    let pi_sum = pi_calc.sum_keepdim(1)?;
    let ones = Tensor::ones((test_in.dim(0)?, 1), candle_core::DType::F32, &device)?;
    let diff_simplex = pi_sum
        .sub(&ones)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_simplex < 1e-5,
        "Simplex invariant error {:.2e} exceeds 1e-5",
        diff_simplex
    );

    println!("Topic 04 Candle track 4-way transitive parity & mathematical invariants verified.");
    Ok(())
}
