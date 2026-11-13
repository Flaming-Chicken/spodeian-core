//! Integration test for Topic 09: Nested, Ragged & Specialized Sparse Tensors (Candle Track).

use candle_core::{Device, Result, safetensors::load};
use sml_candle::c00_foundations::t09_nested_ragged_and_specialized_sparse_tensors::{
    RaggedLinear, csr_spmv, ragged_mean_pool, ragged_to_padded,
};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "00_foundations/09_nested_ragged_and_specialized_sparse_tensors.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let weight_gold = tensors.get("layers.weight").expect("layers.weight");
    let bias_gold = tensors.get("layers.bias").expect("layers.bias");
    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let pooled_gold = tensors.get("layers.pooled_out").expect("layers.pooled_out");
    let dense_padded_gold = tensors
        .get("layers.dense_padded")
        .expect("layers.dense_padded");
    let mask_gold = tensors.get("layers.mask").expect("layers.mask");

    let offsets_tensor = tensors.get("layers.offsets").expect("layers.offsets");
    let offsets_i64 = offsets_tensor.to_vec1::<i64>()?;
    let offsets: Vec<usize> = offsets_i64.into_iter().map(|v| v as usize).collect();

    // 1. Forward Token Projection Parity
    let model = RaggedLinear::new(weight_gold.clone(), bias_gold.clone());
    let token_pred = model.forward(test_in)?;
    let diff_token = token_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_token < 1e-5,
        "Token forward projection error {:.2e} exceeds 1e-5",
        diff_token
    );

    // 2. Ragged Mean Pooling Parity
    let pooled_pred = ragged_mean_pool(&token_pred, &offsets, &device)?;
    let diff_pooled = pooled_pred
        .sub(pooled_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pooled < 1e-5,
        "Ragged mean pooling error {:.2e} exceeds 1e-5",
        diff_pooled
    );

    // 3. CSR SpMV Equivalence Parity
    let crow_i64 = tensors
        .get("layers.csr_crow")
        .expect("csr_crow")
        .to_vec1::<i64>()?;
    let crow: Vec<usize> = crow_i64.into_iter().map(|v| v as usize).collect();
    let col_i64 = tensors
        .get("layers.csr_col")
        .expect("csr_col")
        .to_vec1::<i64>()?;
    let col: Vec<usize> = col_i64.into_iter().map(|v| v as usize).collect();
    let csr_val = tensors
        .get("layers.csr_val")
        .expect("csr_val")
        .to_vec1::<f32>()?;

    let csr_pooled = csr_spmv(
        &crow,
        &col,
        &csr_val,
        &token_pred,
        pooled_gold.dims()[0],
        &device,
    )?;
    let diff_csr = csr_pooled
        .sub(pooled_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_csr < 1e-5,
        "CSR SpMV pooling error {:.2e} exceeds 1e-5",
        diff_csr
    );

    // 4. Ragged-to-Padded Structural Invariant
    let (padded, mask) = ragged_to_padded(test_in, &offsets, &device)?;
    let diff_padded = padded
        .sub(dense_padded_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_mask = mask
        .sub(mask_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_padded < 1e-6,
        "Padded batch error {:.2e} exceeds 1e-6",
        diff_padded
    );
    assert!(
        diff_mask < 1e-6,
        "Padded mask error {:.2e} exceeds 1e-6",
        diff_mask
    );

    Ok(())
}
