use candle_core::{Device, Result, safetensors::load};

use sml_candle::c07_domain_tasks::t01_vision_detection_anchor_free_fcos_and_detr as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_vision_detection_anchor_free_fcos_and_detr_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/01_vision_detection_anchor_free_fcos_and_detr.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // FCOS Centerness
    let centerness = topic::fcos_centerness(&w["ltrb_targets"])?;
    common::assert_max_diff(&centerness, &w["centerness_ref"], 1e-5)?;

    // GIoU
    let giou = topic::generalized_iou(&w["boxes1"], &w["boxes2"])?;
    common::assert_max_diff(&giou, &w["giou_ref"], 1e-5)?;

    // DETR Hungarian matching cost
    let cost = topic::detr_hungarian_cost(
        &w["pred_prob"],
        &w["pred_boxes"],
        &w["gt_classes"],
        &w["gt_boxes"],
        1.0,
        5.0,
        2.0,
    )?;
    common::assert_max_diff(&cost, &w["cost_matrix"], 1e-5)?;

    Ok(())
}
