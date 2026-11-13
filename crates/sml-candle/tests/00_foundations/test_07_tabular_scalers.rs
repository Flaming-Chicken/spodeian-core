//! Integration test for Topic 07: Tabular Scalers, Yeo-Johnson & Weight Folding (Candle Track).

use ai_candle::c00_foundations::t07_feature_engineering_scalers_and_tabular_preprocessing::{
    FusedTabularLinear, StandardScaler, yeo_johnson_transform,
};
use candle_core::{Device, Result, safetensors::load};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let mean_gold = tensors.get("layers.mean").expect("layers.mean");
    let std_gold = tensors.get("layers.std").expect("layers.std");
    let w_head_gold = tensors.get("layers.w_head").expect("layers.w_head");
    let b_head_gold = tensors.get("layers.b_head").expect("layers.b_head");
    let yj_gold = tensors
        .get("layers.yj_reference")
        .expect("layers.yj_reference");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");

    // 1. StandardScaler Invertibility
    let scaler = StandardScaler::new(mean_gold.clone(), std_gold.clone());
    let z = scaler.transform(test_in)?;
    let x_rec = scaler.inverse_transform(&z)?;
    let diff_rec = test_in
        .sub(&x_rec)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rec < 1e-6,
        "Candle scaler inverse transform error {:.2e} exceeds 1e-6",
        diff_rec
    );

    // 2. Yeo-Johnson Transform Parity
    let yj_calc = yeo_johnson_transform(test_in, 0.5)?;
    let diff_yj = yj_calc
        .sub(yj_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_yj < 1e-5,
        "Candle Yeo-Johnson error {:.2e} exceeds 1e-5",
        diff_yj
    );

    // 3. Weight Folding & Forward Evaluation Parity
    let fused =
        FusedTabularLinear::from_scaled_head(mean_gold, std_gold, w_head_gold, b_head_gold)?;
    let diff_w = fused
        .weight
        .sub(weight_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_b = fused
        .bias
        .sub(bias_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_w < 1e-5,
        "Candle fused weight error {:.2e} exceeds 1e-5",
        diff_w
    );
    assert!(
        diff_b < 1e-5,
        "Candle fused bias error {:.2e} exceeds 1e-5",
        diff_b
    );

    let pred = fused.forward(test_in)?;
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

    Ok(())
}
