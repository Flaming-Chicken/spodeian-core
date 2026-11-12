//! Vision Detection Anchor Free FCOS and DETR
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// FCOS Centerness:
/// sqrt( min(l, r)/max(l, r) * min(t, b)/max(t, b) )
pub fn fcos_centerness(ltrb: &Tensor) -> Result<Tensor> {
    let l = ltrb.narrow(2, 0, 1)?;
    let t = ltrb.narrow(2, 1, 1)?;
    let r = ltrb.narrow(2, 2, 1)?;
    let b = ltrb.narrow(2, 3, 1)?;

    let lr_min = l.minimum(&r)?;
    let lr_max = l.maximum(&r)?.clamp(1e-6f32, 1e9f32)?;
    let tb_min = t.minimum(&b)?;
    let tb_max = t.maximum(&b)?.clamp(1e-6f32, 1e9f32)?;

    let lr_term = lr_min.div(&lr_max)?;
    let tb_term = tb_min.div(&tb_max)?;
    let prod = lr_term.mul(&tb_term)?.clamp(0.0f32, 1.0f32)?;
    let sqrt_prod = prod.sqrt()?;

    let (b_sz, n_loc, _) = ltrb.dims3()?;
    sqrt_prod.reshape((b_sz, n_loc))
}

/// Generalized IoU (GIoU) between boxes [N, 4] in [x1, y1, x2, y2] format
pub fn generalized_iou(box1: &Tensor, box2: &Tensor) -> Result<Tensor> {
    let (n, _d) = box1.dims2()?;
    let b1 = box1.to_vec2::<f32>()?;
    let b2 = box2.to_vec2::<f32>()?;

    let mut giou_vec = vec![0f32; n];
    for i in 0..n {
        let x1_i = b1[i][0].max(b2[i][0]);
        let y1_i = b1[i][1].max(b2[i][1]);
        let x2_i = b1[i][2].min(b2[i][2]);
        let y2_i = b1[i][3].min(b2[i][3]);

        let w_i = (x2_i - x1_i).max(0.0f32);
        let h_i = (y2_i - y1_i).max(0.0f32);
        let area_inter = w_i * h_i;

        let area1 = (b1[i][2] - b1[i][0]).max(0.0f32) * (b1[i][3] - b1[i][1]).max(0.0f32);
        let area2 = (b2[i][2] - b2[i][0]).max(0.0f32) * (b2[i][3] - b2[i][1]).max(0.0f32);
        let area_union = area1 + area2 - area_inter;
        let iou = area_inter / area_union.max(1e-6f32);

        let x1_c = b1[i][0].min(b2[i][0]);
        let y1_c = b1[i][1].min(b2[i][1]);
        let x2_c = b1[i][2].max(b2[i][2]);
        let y2_c = b1[i][3].max(b2[i][3]);
        let area_c = (x2_c - x1_c).max(0.0f32) * (y2_c - y1_c).max(0.0f32);

        giou_vec[i] = iou - (area_c - area_union) / area_c.max(1e-6f32);
    }
    Tensor::from_vec(giou_vec, (n,), box1.device())
}

/// DETR Bipartite Matching Cost Matrix:
/// C = cost_class * (-prob[:, gt_classes]) + cost_bbox * L1(pred, gt) + cost_giou * (-GIoU(pred, gt))
pub fn detr_hungarian_cost(
    pred_prob: &Tensor,
    pred_boxes: &Tensor,
    gt_classes: &Tensor,
    gt_boxes: &Tensor,
    cost_class: f64,
    cost_bbox: f64,
    cost_giou: f64,
) -> Result<Tensor> {
    let (n_q, _) = pred_prob.dims2()?;
    let (n_gt, _) = gt_boxes.dims2()?;

    let prob_vec = pred_prob.to_vec2::<f32>()?;
    let classes_vec = gt_classes.to_vec1::<i64>()?;
    let pb_vec = pred_boxes.to_vec2::<f32>()?;
    let gb_vec = gt_boxes.to_vec2::<f32>()?;

    let mut cost_mat = vec![vec![0f32; n_gt]; n_q];

    for i in 0..n_q {
        let p_cx = pb_vec[i][0];
        let p_cy = pb_vec[i][1];
        let p_w = pb_vec[i][2];
        let p_h = pb_vec[i][3];
        let p_x1 = p_cx - 0.5 * p_w;
        let p_y1 = p_cy - 0.5 * p_h;
        let p_x2 = p_cx + 0.5 * p_w;
        let p_y2 = p_cy + 0.5 * p_h;

        for j in 0..n_gt {
            let cls_idx = classes_vec[j] as usize;
            let c_cls = -prob_vec[i][cls_idx];

            let g_cx = gb_vec[j][0];
            let g_cy = gb_vec[j][1];
            let g_w = gb_vec[j][2];
            let g_h = gb_vec[j][3];
            let g_x1 = g_cx - 0.5 * g_w;
            let g_y1 = g_cy - 0.5 * g_h;
            let g_x2 = g_cx + 0.5 * g_w;
            let g_y2 = g_cy + 0.5 * g_h;

            // L1 box distance
            let c_l1 =
                (p_cx - g_cx).abs() + (p_cy - g_cy).abs() + (p_w - g_w).abs() + (p_h - g_h).abs();

            // GIoU
            let x1_i = p_x1.max(g_x1);
            let y1_i = p_y1.max(g_y1);
            let x2_i = p_x2.min(g_x2);
            let y2_i = p_y2.min(g_y2);
            let w_i = (x2_i - x1_i).max(0.0f32);
            let h_i = (y2_i - y1_i).max(0.0f32);
            let area_inter = w_i * h_i;
            let area1 = (p_x2 - p_x1).max(0.0f32) * (p_y2 - p_y1).max(0.0f32);
            let area2 = (g_x2 - g_x1).max(0.0f32) * (g_y2 - g_y1).max(0.0f32);
            let area_union = area1 + area2 - area_inter;
            let iou = area_inter / area_union.max(1e-6f32);

            let x1_c = p_x1.min(g_x1);
            let y1_c = p_y1.min(g_y1);
            let x2_c = p_x2.max(g_x2);
            let y2_c = p_y2.max(g_y2);
            let area_c = (x2_c - x1_c).max(0.0f32) * (y2_c - y1_c).max(0.0f32);
            let giou = iou - (area_c - area_union) / area_c.max(1e-6f32);

            let total = (cost_class as f32) * c_cls
                + (cost_bbox as f32) * c_l1
                + (cost_giou as f32) * (-giou);
            cost_mat[i][j] = total;
        }
    }

    let mut flat = vec![0f32; n_q * n_gt];
    for i in 0..n_q {
        for j in 0..n_gt {
            flat[i * n_gt + j] = cost_mat[i][j];
        }
    }
    Tensor::from_vec(flat, (n_q, n_gt), pred_prob.device())
}
