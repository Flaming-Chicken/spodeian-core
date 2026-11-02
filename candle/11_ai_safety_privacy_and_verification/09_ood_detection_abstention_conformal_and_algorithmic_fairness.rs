//! OOD Detection Abstention Conformal and Algorithmic Fairness
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helmholtz free energy OOD score (Liu et al. 2020): E(x) = -T * logsumexp(logits / T).
pub fn energy_ood_score(logits: &Tensor, temperature: f32) -> Result<Tensor> {
    let l_vec = logits.to_vec2::<f32>()?;
    let b = l_vec.len();
    let c = l_vec[0].len();

    let mut energies = vec![0.0f32; b];
    for i in 0..b {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..c {
            let val = l_vec[i][j] / temperature;
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for j in 0..c {
            sum_exp += ((l_vec[i][j] / temperature) - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        energies[i] = -temperature * lse;
    }
    Tensor::new(energies, logits.device())
}

/// Split Conformal Prediction (Vovk et al. 2005) 1 - alpha quantile.
pub fn split_conformal_quantile(cal_scores: &Tensor, alpha: f32) -> Result<Tensor> {
    let mut scores = cal_scores.flatten_all()?.to_vec1::<f32>()?;
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = scores.len();
    let target = (((n + 1) as f32) * (1.0 - alpha)).ceil() as usize;
    let idx = (target.saturating_sub(1)).min(n - 1);
    Tensor::new(scores[idx], cal_scores.device())
}

/// SelectiveNet risk with coverage penalty constraint (Geifman & El-Yaniv 2019).
pub fn selective_net_loss(
    ce_losses: &Tensor,
    selection_probs: &Tensor,
    target_coverage: f32,
    penalty_weight: f32,
) -> Result<Tensor> {
    let ce = ce_losses.to_vec1::<f32>()?;
    let sp = selection_probs.to_vec1::<f32>()?;
    let b = ce.len();

    let mut sum_sel_risk = 0.0f32;
    let mut sum_sp = 0.0f32;
    for i in 0..b {
        sum_sel_risk += ce[i] * sp[i];
        sum_sp += sp[i];
    }
    let emp_risk = sum_sel_risk / (sum_sp + 1e-8f32);
    let phi_hat = sum_sp / (b as f32);
    let deficit = (target_coverage - phi_hat).max(0.0f32);
    let penalty = penalty_weight * (deficit * deficit);
    Tensor::new(emp_risk + penalty, ce_losses.device())
}

/// Equalized Odds discrepancy (Hardt et al. 2016): |TPR_0 - TPR_1| + |FPR_0 - FPR_1|.
pub fn equalized_odds_discrepancy(
    y_pred: &Tensor,
    y_true: &Tensor,
    sensitive_attr: &Tensor,
) -> Result<Tensor> {
    let yp = y_pred.to_vec1::<f32>()?;
    let yt = y_true.to_vec1::<f32>()?;
    let sa = sensitive_attr.to_vec1::<f32>()?;
    let n = yp.len();

    let mut count_a0_y1 = 0.0f32;
    let mut pred_a0_y1 = 0.0f32;
    let mut count_a1_y1 = 0.0f32;
    let mut pred_a1_y1 = 0.0f32;

    let mut count_a0_y0 = 0.0f32;
    let mut pred_a0_y0 = 0.0f32;
    let mut count_a1_y0 = 0.0f32;
    let mut pred_a1_y0 = 0.0f32;

    for i in 0..n {
        if sa[i] == 0.0 {
            if yt[i] == 1.0 {
                count_a0_y1 += 1.0;
                pred_a0_y1 += yp[i];
            } else {
                count_a0_y0 += 1.0;
                pred_a0_y0 += yp[i];
            }
        } else {
            if yt[i] == 1.0 {
                count_a1_y1 += 1.0;
                pred_a1_y1 += yp[i];
            } else {
                count_a1_y0 += 1.0;
                pred_a1_y0 += yp[i];
            }
        }
    }

    let tpr_0 = pred_a0_y1 / (count_a0_y1 + 1e-8f32);
    let tpr_1 = pred_a1_y1 / (count_a1_y1 + 1e-8f32);
    let fpr_0 = pred_a0_y0 / (count_a0_y0 + 1e-8f32);
    let fpr_1 = pred_a1_y0 / (count_a1_y0 + 1e-8f32);

    let disc = (tpr_0 - tpr_1).abs() + (fpr_0 - fpr_1).abs();
    Tensor::new(disc, y_pred.device())
}
