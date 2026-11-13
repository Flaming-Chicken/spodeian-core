//! Integration test for Category 03 Topic 03: Weight Initialization and muP Scaling (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t03_weight_initialization_and_mup_scaling::{
    LinearAnchor, layerscale_forward, mup_readout_forward, residual_scale_forward, rezero_forward,
};
use candle_core::{Device, Result, safetensors::load};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/03_weight_initialization_and_mup_scaling.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");
    let res_branch = tensors.get("residual_branch").expect("residual_branch");
    let ls_gamma = tensors.get("layerscale_gamma").expect("layerscale_gamma");
    let rz_alpha = tensors.get("rezero_alpha").expect("rezero_alpha");
    let res_scale = tensors
        .get("residual_scale")
        .expect("residual_scale")
        .to_vec1::<f32>()?[0] as f64;
    let x_hid = tensors.get("x_hidden").expect("x_hidden");
    let w_mup_ro = tensors.get("w_mup_readout").expect("w_mup_readout");
    let width_scale = tensors
        .get("width_scale")
        .expect("width_scale")
        .to_vec1::<f32>()?[0] as f64;

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

    // 2. LayerScale Parity
    let y_ls_gold = tensors.get("y_layerscale").expect("y_layerscale");
    let ls_pred = layerscale_forward(test_in, res_branch, ls_gamma)?;
    let diff_ls = ls_pred
        .sub(y_ls_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ls < 1e-5,
        "LayerScale parity error {:.2e} exceeds 1e-5",
        diff_ls
    );

    // 3. ReZero Parity
    let y_rz_gold = tensors.get("y_rezero").expect("y_rezero");
    let rz_pred = rezero_forward(test_in, res_branch, rz_alpha)?;
    let diff_rz = rz_pred
        .sub(y_rz_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rz < 1e-5,
        "ReZero parity error {:.2e} exceeds 1e-5",
        diff_rz
    );

    // 4. Residual Scaling Parity
    let y_scaled_gold = tensors.get("y_residual_scaled").expect("y_residual_scaled");
    let scaled_pred = residual_scale_forward(test_in, res_branch, res_scale)?;
    let diff_scaled = scaled_pred
        .sub(y_scaled_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_scaled < 1e-5,
        "Residual scaling parity error {:.2e} exceeds 1e-5",
        diff_scaled
    );

    // 5. muP Readout Parity
    let y_ro_gold = tensors.get("y_mup_readout").expect("y_mup_readout");
    let ro_pred = mup_readout_forward(x_hid, w_mup_ro, width_scale)?;
    let diff_ro = ro_pred
        .sub(y_ro_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ro < 1e-5,
        "muP Readout parity error {:.2e} exceeds 1e-5",
        diff_ro
    );

    Ok(())
}
