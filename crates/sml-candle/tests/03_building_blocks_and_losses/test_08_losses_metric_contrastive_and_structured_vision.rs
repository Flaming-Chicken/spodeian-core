//! Integration test for Category 03 Topic 08: Losses Metric, Contrastive, and Structured Vision (Candle Track).

use sml_candle::c03_building_blocks_and_losses::t08_losses_metric_contrastive_and_structured_vision::{
    arcface_loss_forward, giou_loss_forward, infonce_loss_forward, siglip_loss_forward,
    soft_dice_loss_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/08_losses_metric_contrastive_and_structured_vision.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let q_repr = tensors.get("q_repr").expect("q_repr");
    let k_repr = tensors.get("k_repr").expect("k_repr");
    let tau = tensors.get("tau_val").expect("tau_val").to_vec1::<f32>()?[0] as f64;
    let loss_infonce_gold = tensors
        .get("loss_infonce")
        .expect("loss_infonce")
        .to_vec1::<f32>()?[0];

    let siglip_scale = tensors
        .get("siglip_scale")
        .expect("siglip_scale")
        .to_vec1::<f32>()?[0] as f64;
    let siglip_bias = tensors
        .get("siglip_bias")
        .expect("siglip_bias")
        .to_vec1::<f32>()?[0] as f64;
    let loss_siglip_gold = tensors
        .get("loss_siglip")
        .expect("loss_siglip")
        .to_vec1::<f32>()?[0];

    let arc_emb = tensors.get("arc_emb").expect("arc_emb");
    let arc_w = tensors.get("arc_weights").expect("arc_weights");
    let arc_targets = tensors.get("arc_targets").expect("arc_targets");
    let arc_s = tensors.get("arc_s").expect("arc_s").to_vec1::<f32>()?[0] as f64;
    let arc_m = tensors.get("arc_m").expect("arc_m").to_vec1::<f32>()?[0] as f64;
    let loss_arcface_gold = tensors
        .get("loss_arcface")
        .expect("loss_arcface")
        .to_vec1::<f32>()?[0];

    let dice_probs = tensors.get("dice_probs").expect("dice_probs");
    let dice_targets = tensors.get("dice_targets").expect("dice_targets");
    let loss_dice_gold = tensors
        .get("loss_dice")
        .expect("loss_dice")
        .to_vec1::<f32>()?[0];

    let boxes1 = tensors.get("boxes1").expect("boxes1");
    let boxes2 = tensors.get("boxes2").expect("boxes2");
    let loss_giou_gold = tensors
        .get("loss_giou")
        .expect("loss_giou")
        .to_vec1::<f32>()?[0];

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

    // 2. InfoNCE Loss Parity
    let loss_infonce_calc = infonce_loss_forward(q_repr, k_repr, tau)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_infonce = (loss_infonce_calc - loss_infonce_gold).abs();
    assert!(
        diff_infonce < 1e-5,
        "InfoNCE parity error {:.2e} exceeds 1e-5",
        diff_infonce
    );

    // 3. SigLIP Loss Parity
    let loss_siglip_calc = siglip_loss_forward(q_repr, k_repr, siglip_scale, siglip_bias)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_siglip = (loss_siglip_calc - loss_siglip_gold).abs();
    assert!(
        diff_siglip < 1e-5,
        "SigLIP parity error {:.2e} exceeds 1e-5",
        diff_siglip
    );

    // 4. ArcFace Loss Parity
    let loss_arc_calc = arcface_loss_forward(arc_emb, arc_w, arc_targets, arc_s, arc_m)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_arc = (loss_arc_calc - loss_arcface_gold).abs();
    assert!(
        diff_arc < 1e-5,
        "ArcFace parity error {:.2e} exceeds 1e-5",
        diff_arc
    );

    // 5. Soft-Dice Loss Parity
    let loss_dice_calc = soft_dice_loss_forward(dice_probs, dice_targets, 1e-6)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_dice = (loss_dice_calc - loss_dice_gold).abs();
    assert!(
        diff_dice < 1e-5,
        "Soft-Dice parity error {:.2e} exceeds 1e-5",
        diff_dice
    );

    // 6. GIoU Loss Parity
    let loss_giou_calc = giou_loss_forward(boxes1, boxes2)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_giou = (loss_giou_calc - loss_giou_gold).abs();
    assert!(
        diff_giou < 1e-5,
        "GIoU parity error {:.2e} exceeds 1e-5",
        diff_giou
    );

    Ok(())
}
