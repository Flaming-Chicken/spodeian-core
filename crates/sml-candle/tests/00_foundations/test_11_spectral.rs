//! Integration test for Topic 11: Randomized Linear Algebra, Spectral Autograd & Hutch++ (Candle Track).

use sml_candle::c00_foundations::t11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp::{
    hutch_plus_plus_trace, lorentzian_broadened_f_matrix, modified_gram_schmidt_qr,
    power_iteration_spectral_norm, SpectralLinear,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");

    let matrix_a = tensors.get("layers.matrix_a").expect("layers.matrix_a");
    let s_sketch = tensors.get("layers.s_sketch").expect("layers.s_sketch");
    let g_rademacher = tensors
        .get("layers.g_rademacher")
        .expect("layers.g_rademacher");
    let q_gold = tensors.get("layers.q_basis").expect("layers.q_basis");
    let r_gold = tensors.get("layers.r_basis").expect("layers.r_basis");
    let hutch_gold = tensors
        .get("layers.hutch_trace_estimate")
        .expect("layers.hutch_trace_estimate");
    let v_init = tensors.get("layers.v_init").expect("layers.v_init");
    let sigma_gold = tensors
        .get("layers.spectral_sigma")
        .expect("layers.spectral_sigma");
    let colliding_sigma = tensors
        .get("layers.colliding_sigma")
        .expect("layers.colliding_sigma");
    let lorentzian_f_gold = tensors
        .get("layers.lorentzian_f")
        .expect("layers.lorentzian_f");

    // 1. Modified Gram-Schmidt QR Parity on A @ S
    let y_sketch = matrix_a.matmul(s_sketch)?;
    let (q_pred, r_pred) = modified_gram_schmidt_qr(&y_sketch, &device)?;
    let diff_q = q_pred
        .sub(q_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_r = r_pred
        .sub(r_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_q < 1e-5,
        "MGS Q basis error {:.2e} exceeds 1e-5",
        diff_q
    );
    assert!(
        diff_r < 1e-5,
        "MGS R basis error {:.2e} exceeds 1e-5",
        diff_r
    );

    // 2. Hutch++ Stochastic Trace Estimator Parity
    let hutch_pred = hutch_plus_plus_trace(matrix_a, s_sketch, g_rademacher, &device)?;
    let diff_hutch = hutch_pred
        .sub(hutch_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_hutch < 1e-5,
        "Hutch++ trace estimate error {:.2e} exceeds 1e-5",
        diff_hutch
    );

    // 3. Power Iteration Spectral Norm Parity
    let (sigma_pred, _, _) = power_iteration_spectral_norm(weight_gold, v_init, 5, &device)?;
    let diff_sigma = sigma_pred
        .sub(sigma_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sigma < 1e-5,
        "Power iteration spectral norm error {:.2e} exceeds 1e-5",
        diff_sigma
    );

    // 4. Lorentzian-Broadened Spectral Autograd Coupling Matrix Parity
    let f_pred = lorentzian_broadened_f_matrix(colliding_sigma, 0.01, &device)?;
    let diff_f = f_pred
        .sub(lorentzian_f_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_f < 1e-5,
        "Lorentzian F matrix error {:.2e} exceeds 1e-5",
        diff_f
    );

    // 5. Model Forward Projection Parity
    let model = SpectralLinear::new(weight_gold.clone(), bias_gold.clone());
    let pred = model.forward(test_in)?;
    let diff_pred = pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pred < 1e-5,
        "SpectralLinear forward error {:.2e} exceeds 1e-5",
        diff_pred
    );

    Ok(())
}
