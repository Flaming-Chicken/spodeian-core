//! PyTorch 2 Parity Loss Functions for `sml-runtime`.
//!
//! Provides mathematically exact, numerically conditioned loss functions:
//! - CrossEntropyLoss (with Label Smoothing & Multi-Dimensional Reduction)
//! - BCEWithLogitsLoss (Log-Sum-Exp Numerically Stable Formulation)
//! - MSELoss, L1Loss, SmoothL1Loss (Huber)
//! - FocalLoss (Lin et al. Imbalanced Classification)
//! - InfoNCELoss (Contrastive Representation Learning)
//! - VICRegLoss (Variance-Invariance-Covariance Representation Regularization)
//! - FlowMatchingLoss (Optimal Transport Velocity Vector Field Regression)

use candle_core::{Result, Tensor};

/// Reduction mode for loss functions matching PyTorch 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

impl Default for Reduction {
    fn default() -> Self {
        Self::Mean
    }
}

fn apply_reduction(tensor: &Tensor, reduction: Reduction) -> Result<Tensor> {
    match reduction {
        Reduction::None => Ok(tensor.clone()),
        Reduction::Mean => tensor.mean_all(),
        Reduction::Sum => tensor.sum_all(),
    }
}

// ============================================================================
// 1. CrossEntropyLoss
// ============================================================================

/// Cross-entropy loss with optional label smoothing.
///
/// Logits shape: `[B, C]` or `[B, T, C]`.
/// Targets shape: `[B]` or `[B, T]` (class indices).
pub fn cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    label_smoothing: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    let dims = logits.dims();
    let num_classes = *dims.last().unwrap();

    // Flatten leading dimensions if sequence or spatial: e.g. [B, T, C] -> [B*T, C]
    let (flat_logits, flat_targets) = if dims.len() > 2 {
        let total_samples: usize = dims[..dims.len() - 1].iter().product();
        let flat_l = logits.reshape((total_samples, num_classes))?;
        let flat_t = targets.reshape((total_samples,))?;
        (flat_l, flat_t)
    } else {
        (logits.clone(), targets.clone())
    };

    let log_probs = candle_nn::ops::log_softmax(&flat_logits, 1)?;

    if label_smoothing <= 0.0 {
        // Standard NLL loss: -log_prob[target]
        let target_log_probs = log_probs.gather(&flat_targets.unsqueeze(1)?, 1)?.squeeze(1)?;
        let nll = target_log_probs.neg()?;
        apply_reduction(&nll, reduction)
    } else {
        // Label smoothing: (1 - eps) * nll + eps * mean(-log_probs)
        let target_log_probs = log_probs.gather(&flat_targets.unsqueeze(1)?, 1)?.squeeze(1)?;
        let nll = target_log_probs.neg()?;
        let smooth_loss = log_probs.mean_keepdim(1)?.squeeze(1)?.neg()?;
        let total = ((nll * (1.0 - label_smoothing))? + (smooth_loss * label_smoothing)?)?;
        apply_reduction(&total, reduction)
    }
}

// ============================================================================
// 2. BCEWithLogitsLoss
// ============================================================================

/// Numerically stable Binary Cross-Entropy with Logits:
/// loss = max(x, 0) - x * z + log(1 + exp(-|x|))
pub fn bce_with_logits(
    logits: &Tensor,
    targets: &Tensor,
    reduction: Reduction,
) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(logits)?;
    let max_val = logits.maximum(&zeros)?;
    let abs_logits = logits.abs()?;
    let neg_abs = abs_logits.neg()?;
    let log_term = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;

    // max_val - logits * targets + log_term
    let x_z = logits.mul(targets)?;
    let loss = max_val.sub(&x_z)?.add(&log_term)?;
    apply_reduction(&loss, reduction)
}

// ============================================================================
// 3. MSE, L1, SmoothL1 (Huber) Losses
// ============================================================================

/// Mean Squared Error loss: (pred - target)^2
pub fn mse_loss(pred: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    let diff = pred.sub(target)?;
    let sqr = diff.sqr()?;
    apply_reduction(&sqr, reduction)
}

/// L1 / Mean Absolute Error loss: |pred - target|
pub fn l1_loss(pred: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    let diff = pred.sub(target)?;
    let abs = diff.abs()?;
    apply_reduction(&abs, reduction)
}

/// Smooth L1 / Huber Loss with transition threshold beta:
/// if |x| < beta => 0.5 * x^2 / beta
/// else => |x| - 0.5 * beta
pub fn smooth_l1_loss(
    pred: &Tensor,
    target: &Tensor,
    beta: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    let diff = pred.sub(target)?.abs()?;

    // Condition: diff < beta
    let quadratic = (diff.sqr()?.affine(0.5 / beta, 0.0))?;
    let linear = (diff.affine(1.0, -0.5 * beta))?;

    // Smooth selection: min(quadratic, linear)
    let loss = quadratic.minimum(&linear)?;
    apply_reduction(&loss, reduction)
}

// ============================================================================
// 4. FocalLoss
// ============================================================================

/// Focal Loss for imbalanced classification:
/// FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
pub fn focal_loss(
    logits: &Tensor,
    targets: &Tensor,
    gamma: f64,
    alpha: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
    let probs = candle_nn::ops::softmax(logits, 1)?;

    let target_log_probs = log_probs.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;
    let target_probs = probs.gather(&targets.unsqueeze(1)?, 1)?.squeeze(1)?;

    let one_minus_p = target_probs.affine(-1.0, 1.0)?;
    let focal_weight = if gamma == 0.0 {
        Tensor::ones_like(&one_minus_p)?
    } else {
        one_minus_p.powf(gamma)?
    };

    let loss = (target_log_probs.neg()?.mul(&focal_weight)? * alpha)?;
    apply_reduction(&loss, reduction)
}

// ============================================================================
// 5. InfoNCELoss
// ============================================================================

/// InfoNCE contrastive loss:
/// L = -log( exp(q . k+ / tau) / [exp(q . k+ / tau) + sum_j exp(q . k-_j / tau)] )
/// Queries: `[B, D]`, Positives: `[B, D]`, Temperature: `tau`.
pub fn info_nce_loss(queries: &Tensor, positives: &Tensor, temperature: f64) -> Result<Tensor> {
    let tau = temperature.max(1e-4);
    // Normalized embeddings
    let q_norm = queries.sqr()?.sum_keepdim(1)?.sqrt()?;
    let q = queries.broadcast_div(&q_norm)?;
    let k_norm = positives.sqr()?.sum_keepdim(1)?.sqrt()?;
    let k = positives.broadcast_div(&k_norm)?;

    // Cosine similarity matrix: [B, B]
    let sim_matrix = (q.matmul(&k.t()?)?.affine(1.0 / tau, 0.0))?;
    let b = queries.dim(0)?;
    let targets = Tensor::from_vec((0..b as u32).collect(), (b,), queries.device())?;

    cross_entropy(&sim_matrix, &targets, 0.0, Reduction::Mean)
}

// ============================================================================
// 6. VICRegLoss (Variance-Invariance-Covariance)
// ============================================================================

/// VICReg self-supervised regularization loss (Bardes et al., 2022).
pub fn vicreg_loss(
    z_a: &Tensor,
    z_b: &Tensor,
    sim_coeff: f64,
    var_coeff: f64,
    cov_coeff: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (b, d) = z_a.dims2()?;

    // 1. Invariance MSE loss
    let sim_loss = z_a.sub(z_b)?.sqr()?.mean_all()?;

    // 2. Variance anti-collapse loss: hinge on std >= 1.0
    let std_loss = |z: &Tensor| -> Result<Tensor> {
        let mean = z.mean_keepdim(0)?;
        let cent = z.broadcast_sub(&mean)?;
        let var = cent.sqr()?.mean_keepdim(0)?;
        let std = var.affine(1.0, 1e-4)?.sqrt()?;
        let hinge = std.affine(-1.0, 1.0)?.maximum(&Tensor::zeros_like(&std)?)?;
        hinge.mean_all()
    };
    let var_loss = ((std_loss(z_a)? + std_loss(z_b)?)?.affine(0.5, 0.0))?;

    // 3. Covariance decorrelation loss (penalize off-diagonal terms)
    let cov_loss_fn = |z: &Tensor| -> Result<Tensor> {
        let mean = z.mean_keepdim(0)?;
        let cent = z.broadcast_sub(&mean)?;
        let cov = (cent.t()?.matmul(&cent)?.affine(1.0 / (b - 1).max(1) as f64, 0.0))?;
        let cov_sqr = cov.sqr()?;
        // Total sum minus diagonal sum
        let total_sq = cov_sqr.sum_all()?.to_scalar::<f32>()? as f64;
        let mut diag_sq = 0.0f64;
        let cov_vec = cov_sqr.to_vec2::<f32>()?;
        for i in 0..d {
            diag_sq += cov_vec[i][i] as f64;
        }
        let off_diag = ((total_sq - diag_sq).max(0.0) / d as f64) as f32;
        Tensor::new(off_diag, z.device())
    };
    let cov_loss = ((cov_loss_fn(z_a)? + cov_loss_fn(z_b)?)?.affine(0.5, 0.0))?;

    // Combined loss
    let total = (((sim_loss.clone() * sim_coeff)? + (var_loss.clone() * var_coeff)?)? + (cov_loss.clone() * cov_coeff)?)?;
    Ok((total, sim_loss, var_loss, cov_loss))
}

// ============================================================================
// 7. FlowMatchingLoss
// ============================================================================

/// Optimal Transport Continuous Flow Matching (OT-CFM) regression loss:
/// L = ||v_theta(x_t, t) - (x_1 - x_0)||^2
pub fn flow_matching_loss(pred_v: &Tensor, target_v: &Tensor) -> Result<Tensor> {
    mse_loss(pred_v, target_v, Reduction::Mean)
}
