//! Integration test for Topic 05: Hooks & Saved Tensor Hooks (Candle Track).

use candle_core::{Device, Result, safetensors::load};
use sml_candle::c00_foundations::t05_hooks_and_saved_tensor_hooks::HookedLinearLayer;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path =
        common::get_weights_path("00_foundations/05_hooks_and_saved_tensor_hooks.safetensors");
    let tensors = load(weights_path, &device)?;

    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let target = tensors.get("layers.target").expect("layers.target");

    let golden_gw = tensors
        .get("layers.grad_weight")
        .expect("layers.grad_weight");
    let golden_gb = tensors.get("layers.grad_bias").expect("layers.grad_bias");
    let golden_gx = tensors.get("layers.grad_input").expect("layers.grad_input");

    let layer = HookedLinearLayer::new(weight.clone(), bias.clone());

    // 1. Forward pass with activation interception
    let (pred, saved_x) = layer.forward_with_intercept(test_in)?;

    // 2. Parity on forward output
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

    // 3. Backward pass from intercepted activations
    let (gx, gw, gb) = layer.backward_from_intercept(&saved_x, &pred, target)?;

    let diff_gw = gw
        .sub(golden_gw)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_gb = gb
        .sub(golden_gb)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_gx = gx
        .sub(golden_gx)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(
        diff_gw < 1e-4,
        "Candle grad_weight error {:.2e} exceeds 1e-4",
        diff_gw
    );
    assert!(
        diff_gb < 1e-4,
        "Candle grad_bias error {:.2e} exceeds 1e-4",
        diff_gb
    );
    assert!(
        diff_gx < 1e-4,
        "Candle grad_input error {:.2e} exceeds 1e-4",
        diff_gx
    );

    println!(
        "Candle Topic 05 Parity: Pred err={:.2e}, GW err={:.2e}, GB err={:.2e}, GX err={:.2e}",
        diff_pred, diff_gw, diff_gb, diff_gx
    );

    Ok(())
}
