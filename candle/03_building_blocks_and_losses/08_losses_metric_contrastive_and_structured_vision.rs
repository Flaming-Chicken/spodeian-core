//! Losses: Metric, Contrastive, and Structured Vision (C03 Topic 08).
//!
//! Native Rust Candle implementation of:
//! 1. InfoNCE Loss (Oord et al. 2018).
//! 2. SigLIP Loss (Zhai et al. 2023).
//! 3. ArcFace Loss (Deng et al. 2019).
//! 4. Soft-Dice Loss (Milletari et al. 2016).
//! 5. Generalized IoU (GIoU) Loss (Rezatofighi et al. 2019).
//! 6. Linear Anchor.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// Helper for L2 normalization along the last dimension.
fn l2_normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let sum_sq = sq.sum_keepdim(1)?;
    let norm = (sum_sq + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// InfoNCE Loss forward with L2 normalization.
pub fn infonce_loss_forward(q: &Tensor, k: &Tensor, tau: f64) -> Result<Tensor> {
    let q_norm = l2_normalize(q, 1e-12)?;
    let k_norm = l2_normalize(k, 1e-12)?;
    let k_t = k_norm.t()?;
    let sim = q_norm.matmul(&k_t)?;
    let logits = (sim / tau)?;

    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;
    let b = log_probs_vec.len();

    let mut total = 0.0f32;
    for i in 0..b {
        total += -log_probs_vec[i][i];
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], q.device())
}

/// SigLIP Loss forward: pairwise sigmoid binary cross-entropy.
pub fn siglip_loss_forward(q: &Tensor, k: &Tensor, scale: f64, bias: f64) -> Result<Tensor> {
    let q_norm = l2_normalize(q, 1e-12)?;
    let k_norm = l2_normalize(k, 1e-12)?;
    let k_t = k_norm.t()?;
    let sim = q_norm.matmul(&k_t)?;
    let logits = ((sim * scale)? + bias)?;

    let logits_vec = logits.to_vec2::<f32>()?;
    let b = logits_vec.len();

    let mut total = 0.0f32;
    for i in 0..b {
        for j in 0..b {
            let label = if i == j { 1.0f32 } else { -1.0f32 };
            let z = -label * logits_vec[i][j];
            // softplus(z) = max(0, z) + ln(1 + exp(-|z|))
            let max_val = z.max(0.0);
            let softplus = max_val + (1.0 + (-z.abs()).exp()).ln();
            total += softplus;
        }
    }
    let mean = total / ((b * b) as f32);
    Tensor::new(&[mean], q.device())
}

/// ArcFace Loss forward: Additive Angular Margin in cos-space.
pub fn arcface_loss_forward(
    embeddings: &Tensor,
    weights: &Tensor,
    targets: &Tensor,
    s: f64,
    m: f64,
) -> Result<Tensor> {
    let emb_norm = l2_normalize(embeddings, 1e-12)?;
    let w_norm = l2_normalize(weights, 1e-12)?;
    let w_t = w_norm.t()?;
    let cos_theta = emb_norm.matmul(&w_t)?;

    let cos_vec = cos_theta.to_vec2::<f32>()?;
    let targets_vec: Vec<usize> = match targets.dtype() {
        candle_core::DType::I64 => targets
            .flatten_all()?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as usize)
            .collect(),
        _ => targets
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as usize)
            .collect(),
    };
    let b = cos_vec.len();
    let num_classes = cos_vec[0].len();

    let cos_m = (m as f32).cos();
    let sin_m = (m as f32).sin();
    let s_f = s as f32;

    let mut logits_vec = vec![vec![0.0f32; num_classes]; b];
    for i in 0..b {
        let target_idx = targets_vec[i];
        for c in 0..num_classes {
            let cos_t = cos_vec[i][c].clamp(-0.9999999, 0.9999999);
            if c == target_idx {
                let sin_t = (1.0 - cos_t * cos_t).max(1e-7).sqrt();
                let cos_t_m = cos_t * cos_m - sin_t * sin_m;
                logits_vec[i][c] = s_f * cos_t_m;
            } else {
                logits_vec[i][c] = s_f * cos_t;
            }
        }
    }

    let logits_flat: Vec<f32> = logits_vec.into_iter().flatten().collect();
    let logits_tensor = Tensor::from_vec(logits_flat, (b, num_classes), embeddings.device())?;
    let log_probs = candle_nn::ops::log_softmax(&logits_tensor, 1)?;
    let log_probs_vec = log_probs.to_vec2::<f32>()?;

    let mut total = 0.0f32;
    for i in 0..b {
        let t = targets_vec[i];
        total += -log_probs_vec[i][t];
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], embeddings.device())
}

/// Soft-Dice Loss forward.
pub fn soft_dice_loss_forward(probs: &Tensor, targets: &Tensor, eps: f64) -> Result<Tensor> {
    let p_vec = probs.flatten_all()?.to_vec1::<f32>()?;
    let t_vec = targets.flatten_all()?.to_vec1::<f32>()?;

    let mut intersection = 0.0f32;
    let mut den = 0.0f32;
    for (&p, &t) in p_vec.iter().zip(t_vec.iter()) {
        intersection += p * t;
        den += p * p + t * t;
    }
    let eps_f = eps as f32;
    let dice = (2.0 * intersection + eps_f) / (den + eps_f);
    let loss = 1.0 - dice;
    Tensor::new(&[loss], probs.device())
}

/// Generalized IoU (GIoU) Loss forward.
pub fn giou_loss_forward(boxes1: &Tensor, boxes2: &Tensor) -> Result<Tensor> {
    let b1 = boxes1.to_vec2::<f32>()?;
    let b2 = boxes2.to_vec2::<f32>()?;
    let n = b1.len();

    let mut total_loss = 0.0f32;
    let eps = 1e-7f32;

    for i in 0..n {
        let x1_i = b1[i][0].max(b2[i][0]);
        let y1_i = b1[i][1].max(b2[i][1]);
        let x2_i = b1[i][2].min(b2[i][2]);
        let y2_i = b1[i][3].min(b2[i][3]);
        let inter_area = (x2_i - x1_i).max(0.0) * (y2_i - y1_i).max(0.0);

        let area1 = (b1[i][2] - b1[i][0]).max(0.0) * (b1[i][3] - b1[i][1]).max(0.0);
        let area2 = (b2[i][2] - b2[i][0]).max(0.0) * (b2[i][3] - b2[i][1]).max(0.0);
        let union_area = area1 + area2 - inter_area + eps;
        let iou = inter_area / union_area;

        let x1_c = b1[i][0].min(b2[i][0]);
        let y1_c = b1[i][1].min(b2[i][1]);
        let x2_c = b1[i][2].max(b2[i][2]);
        let y2_c = b1[i][3].max(b2[i][3]);
        let c_area = (x2_c - x1_c).max(0.0) * (y2_c - y1_c).max(0.0) + eps;

        let giou = iou - (c_area - union_area) / c_area;
        total_loss += 1.0 - giou;
    }

    let mean = total_loss / (n as f32);
    Tensor::new(&[mean], boxes1.device())
}
