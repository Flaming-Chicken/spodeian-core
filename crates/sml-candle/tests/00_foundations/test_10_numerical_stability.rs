//! Integration test for Topic 10: Numerical Stability, Welford Variance & LogSumExp (Candle Track).

use ai_candle::c00_foundations::t10_floating_point_arithmetic_conditioning_and_numerical_stability::{
    neumaier_sum_2d, welford_mean_var, StableLinear,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/10_floating_point_arithmetic_conditioning_and_numerical_stability.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let mean_gold = tensors
        .get("layers.welford_mean")
        .expect("layers.welford_mean");
    let var_gold = tensors
        .get("layers.welford_var")
        .expect("layers.welford_var");
    let lse_gold = tensors.get("layers.lse_ref").expect("layers.lse_ref");

    // 1. Welford Online Statistics Parity
    let (mean_pred, var_pred) = welford_mean_var(test_in, &device)?;
    let diff_mean = mean_pred
        .sub(mean_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_var = var_pred
        .sub(var_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mean < 1e-6,
        "Welford mean error {:.2e} exceeds 1e-6",
        diff_mean
    );
    assert!(
        diff_var < 1e-6,
        "Welford var error {:.2e} exceeds 1e-6",
        diff_var
    );

    // 2. Model Forward Evaluation Parity
    let model = StableLinear::new(weight_gold.clone(), bias_gold.clone());
    let pred = model.forward(test_in)?;
    let diff_pred = pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pred < 1e-5,
        "Candle forward prediction error {:.2e} exceeds 1e-5",
        diff_pred
    );

    // 3. Stable Log-Sum-Exp Evaluation Parity
    let lse_pred = model.forward_lse(test_in, &device)?;
    let diff_lse = lse_pred
        .sub(lse_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_lse < 1e-5,
        "Candle stable LSE error {:.2e} exceeds 1e-5",
        diff_lse
    );

    // 4. Neumaier Compensated Summation
    let n_sum = neumaier_sum_2d(test_in, &device)?;
    assert_eq!(n_sum.dims(), &[test_in.dims()[1]]);

    Ok(())
}
