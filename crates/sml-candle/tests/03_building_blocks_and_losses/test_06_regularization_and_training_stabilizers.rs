//! Integration test for Category 03 Topic 06: Regularization and Training Stabilizers (Candle Track).

use candle_core::{Device, Result, safetensors::load};
use sml_candle::c03_building_blocks_and_losses::t06_regularization_and_training_stabilizers::{
    LinearAnchor, dropout_forward, droppath_forward, label_smoothing_loss_forward,
    router_z_loss_forward, softcapping_forward,
};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/06_regularization_and_training_stabilizers.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let x_drop = tensors.get("x_drop").expect("x_drop");
    let drop_mask = tensors.get("drop_mask").expect("drop_mask");
    let drop_rate = tensors
        .get("drop_rate")
        .expect("drop_rate")
        .to_vec1::<f32>()?[0] as f64;
    let y_drop_gold = tensors.get("y_drop").expect("y_drop");

    let x_id = tensors.get("x_identity").expect("x_identity");
    let x_br = tensors.get("x_branch").expect("x_branch");
    let dp_mask = tensors.get("droppath_mask").expect("droppath_mask");
    let dp_rate = tensors
        .get("droppath_rate")
        .expect("droppath_rate")
        .to_vec1::<f32>()?[0] as f64;
    let y_droppath_gold = tensors.get("y_droppath").expect("y_droppath");

    let x_log = tensors.get("x_logits").expect("x_logits");
    let softcap = tensors
        .get("softcap_val")
        .expect("softcap_val")
        .to_vec1::<f32>()?[0] as f64;
    let y_softcap_gold = tensors.get("y_softcap").expect("y_softcap");

    let logits_ls = tensors.get("logits_ls").expect("logits_ls");
    let targets_ls = tensors.get("targets_ls").expect("targets_ls");
    let alpha_ls = tensors
        .get("smooth_alpha")
        .expect("smooth_alpha")
        .to_vec1::<f32>()?[0] as f64;
    let loss_ls_gold = tensors.get("loss_ls").expect("loss_ls").to_vec1::<f32>()?[0];

    let router_logits = tensors.get("router_logits").expect("router_logits");
    let z_coef = tensors.get("z_coef").expect("z_coef").to_vec1::<f32>()?[0] as f64;
    let loss_z_gold = tensors.get("loss_z").expect("loss_z").to_vec1::<f32>()?[0];

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

    // 2. Inverted Dropout Parity
    let drop_pred = dropout_forward(x_drop, drop_mask, drop_rate)?;
    let diff_drop = drop_pred
        .sub(y_drop_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_drop < 1e-5,
        "Dropout parity error {:.2e} exceeds 1e-5",
        diff_drop
    );

    // 3. DropPath Parity
    let dp_pred = droppath_forward(x_id, x_br, dp_mask, dp_rate)?;
    let diff_dp = dp_pred
        .sub(y_droppath_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_dp < 1e-5,
        "DropPath parity error {:.2e} exceeds 1e-5",
        diff_dp
    );

    // 4. Logit Soft-Capping Parity
    let sc_pred = softcapping_forward(x_log, softcap)?;
    let diff_sc = sc_pred
        .sub(y_softcap_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sc < 1e-5,
        "SoftCap parity error {:.2e} exceeds 1e-5",
        diff_sc
    );

    // 5. Label Smoothing Loss Parity
    let loss_ls_pred = label_smoothing_loss_forward(logits_ls, targets_ls, alpha_ls)?;
    let loss_ls_val = loss_ls_pred.to_vec1::<f32>()?[0];
    let diff_ls = (loss_ls_val - loss_ls_gold).abs();
    assert!(
        diff_ls < 1e-5,
        "Label Smoothing loss error {:.2e} exceeds 1e-5",
        diff_ls
    );

    // 6. Router Z-Loss Parity
    let loss_z_pred = router_z_loss_forward(router_logits, z_coef)?;
    let loss_z_val = loss_z_pred.to_vec1::<f32>()?[0];
    let diff_z = (loss_z_val - loss_z_gold).abs();
    assert!(
        diff_z < 1e-5,
        "Router Z-loss error {:.2e} exceeds 1e-5",
        diff_z
    );

    Ok(())
}
