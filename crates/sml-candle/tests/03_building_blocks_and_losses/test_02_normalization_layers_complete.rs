//! Integration test for Category 03 Topic 02: Normalization Layers Complete (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t02_normalization_layers_complete::{
    adaln_zero_forward, dyt_forward, layernorm_forward, qknorm_forward, rmsnorm_forward,
    weight_standardization_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/02_normalization_layers_complete.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");
    let gamma = tensors.get("gamma").expect("gamma");
    let beta = tensors.get("beta").expect("beta");
    let dyt_alpha = tensors.get("dyt_alpha").expect("dyt_alpha");
    let dyt_gamma = tensors.get("dyt_gamma").expect("dyt_gamma");
    let dyt_beta = tensors.get("dyt_beta").expect("dyt_beta");
    let cond = tensors.get("cond_vector").expect("cond_vector");
    let w_adaln = tensors.get("w_adaln").expect("w_adaln");
    let b_adaln = tensors.get("b_adaln").expect("b_adaln");
    let w_mat = tensors.get("w_matrix").expect("w_matrix");
    let eps_val = tensors.get("eps").expect("eps").to_vec1::<f32>()?[0] as f64;

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

    // 2. LayerNorm Parity
    let y_ln_gold = tensors.get("y_layernorm").expect("y_layernorm");
    let ln_pred = layernorm_forward(test_in, gamma, beta, eps_val)?;
    let diff_ln = ln_pred
        .sub(y_ln_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ln < 1e-5,
        "LayerNorm parity error {:.2e} exceeds 1e-5",
        diff_ln
    );

    // 3. RMSNorm Parity
    let y_rms_gold = tensors.get("y_rmsnorm").expect("y_rmsnorm");
    let rms_pred = rmsnorm_forward(test_in, gamma, eps_val)?;
    let diff_rms = rms_pred
        .sub(y_rms_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rms < 1e-5,
        "RMSNorm parity error {:.2e} exceeds 1e-5",
        diff_rms
    );

    // 4. Dynamic Tanh (DyT) Parity
    let y_dyt_gold = tensors.get("y_dyt").expect("y_dyt");
    let dyt_pred = dyt_forward(test_in, dyt_alpha, dyt_gamma, dyt_beta)?;
    let diff_dyt = dyt_pred
        .sub(y_dyt_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_dyt < 1e-5,
        "DyT parity error {:.2e} exceeds 1e-5",
        diff_dyt
    );

    // 5. adaLN-Zero Parity
    let y_adaln_gold = tensors.get("y_adaln_zero").expect("y_adaln_zero");
    let adaln_pred = adaln_zero_forward(test_in, cond, w_adaln, b_adaln, eps_val)?;
    let diff_adaln = adaln_pred
        .sub(y_adaln_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_adaln < 1e-5,
        "adaLN-Zero parity error {:.2e} exceeds 1e-5",
        diff_adaln
    );

    // 6. QK-Norm Parity
    let y_qk_gold = tensors.get("y_qknorm").expect("y_qknorm");
    let qk_pred = qknorm_forward(test_in, eps_val)?;
    let diff_qk = qk_pred
        .sub(y_qk_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_qk < 1e-5,
        "QK-Norm parity error {:.2e} exceeds 1e-5",
        diff_qk
    );

    // 7. Scaled Weight Standardization Parity
    let w_std_gold = tensors.get("w_standardized").expect("w_standardized");
    let w_std_pred = weight_standardization_forward(w_mat, eps_val)?;
    let diff_ws = w_std_pred
        .sub(w_std_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ws < 1e-5,
        "Weight Standardization parity error {:.2e} exceeds 1e-5",
        diff_ws
    );

    Ok(())
}
