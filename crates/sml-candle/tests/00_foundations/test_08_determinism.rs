//! Integration test for Topic 08: Determinism, Kahan Summation & Hash Partitioning (Candle Track).

use ai_candle::c00_foundations::t08_reproducible_preprocessing_pipelines_and_determinism::{
    fnv1a_hash, hash_deterministic_split, kahan_sum_2d, DeterministicLinear,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/08_reproducible_preprocessing_pipelines_and_determinism.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let kahan_mean_gold = tensors.get("layers.kahan_mean").expect("layers.kahan_mean");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");

    // 1. Kahan Summation Parity
    let k_sum = kahan_sum_2d(test_in)?;
    let k_mean = (k_sum / (test_in.dims()[0] as f64))?;
    let diff_kahan = k_mean
        .sub(kahan_mean_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_kahan < 1e-6,
        "Candle Kahan mean error {:.2e} exceeds 1e-6",
        diff_kahan
    );

    // 2. FNV-1a Hash Partition Invariant
    let s1 = hash_deterministic_split("sample_42", "experiment_alpha", 0.8);
    let s2 = hash_deterministic_split("sample_42", "experiment_alpha", 0.8);
    assert_eq!(s1, s2);
    assert_eq!(fnv1a_hash("test", "salt"), fnv1a_hash("test", "salt"));

    // 3. Model Forward Evaluation Parity
    let model = DeterministicLinear::new(weight_gold.clone(), bias_gold.clone());
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

    Ok(())
}
