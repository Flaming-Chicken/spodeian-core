//! Machine Unlearning SISA Influence Scrub NPO and RMU
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Numerically stable softplus: ln(1 + exp(x)).
fn softplus(x: f32) -> f32 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Negative Preference Optimization (NPO) loss for machine unlearning (Zhang et al. 2024).
pub fn npo_loss(
    logits_theta: &Tensor,
    logits_ref: &Tensor,
    labels: &Tensor,
    beta: f32,
) -> Result<Tensor> {
    let l_th = logits_theta.to_vec2::<f32>()?;
    let l_rf = logits_ref.to_vec2::<f32>()?;
    let labs = labels.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;

    let b = l_th.len();
    let c = l_th[0].len();

    let mut total_loss = 0.0f32;
    for i in 0..b {
        let y = labs[i] as usize;

        // Log-sum-exp theta
        let mut max_th = l_th[i][0];
        for k in 1..c {
            if l_th[i][k] > max_th {
                max_th = l_th[i][k];
            }
        }
        let mut sum_exp_th = 0.0f32;
        for k in 0..c {
            sum_exp_th += (l_th[i][k] - max_th).exp();
        }
        let lse_th = max_th + sum_exp_th.ln();
        let log_pi_th = l_th[i][y] - lse_th;

        // Log-sum-exp ref
        let mut max_rf = l_rf[i][0];
        for k in 1..c {
            if l_rf[i][k] > max_rf {
                max_rf = l_rf[i][k];
            }
        }
        let mut sum_exp_rf = 0.0f32;
        for k in 0..c {
            sum_exp_rf += (l_rf[i][k] - max_rf).exp();
        }
        let lse_rf = max_rf + sum_exp_rf.ln();
        let log_pi_rf = l_rf[i][y] - lse_rf;

        let log_ratio = log_pi_th - log_pi_rf;
        let sample_loss = (2.0 / beta) * softplus(beta * log_ratio);
        total_loss += sample_loss;
    }

    Tensor::new(total_loss / (b as f32), logits_theta.device())
}

/// Representation Misdirection for Unlearning (RMU — Li et al. 2024).
pub fn rmu_loss(
    h_theta_forget: &Tensor,
    h_target_random: &Tensor,
    h_theta_retain: &Tensor,
    h_ref_retain: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    let diff_f = h_theta_forget.sub(h_target_random)?;
    let l_forget = diff_f.sqr()?.mean_all()?;

    let diff_r = h_theta_retain.sub(h_ref_retain)?;
    let l_retain = diff_r.sqr()?.mean_all()?;

    let scaled_retain = (l_retain * (alpha as f64))?;
    l_forget.add(&scaled_retain)
}

/// SCRUB student-teacher unlearning loss (Kurmanji et al. 2023).
pub fn scrub_loss(
    logits_f_theta: &Tensor,
    labels_f: &Tensor,
    logits_r_theta: &Tensor,
    logits_r_ref: &Tensor,
    gamma: f32,
    tau: f32,
) -> Result<Tensor> {
    let lf_vec = logits_f_theta.to_vec2::<f32>()?;
    let yf_vec = labels_f
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()?;
    let lr_th = logits_r_theta.to_vec2::<f32>()?;
    let lr_rf = logits_r_ref.to_vec2::<f32>()?;

    let bf = lf_vec.len();
    let cf = lf_vec[0].len();
    let br = lr_th.len();
    let cr = lr_th[0].len();

    // 1. Forget cross entropy
    let mut total_ce_f = 0.0f32;
    for i in 0..bf {
        let y = yf_vec[i] as usize;
        let mut max_val = lf_vec[i][0];
        for k in 1..cf {
            if lf_vec[i][k] > max_val {
                max_val = lf_vec[i][k];
            }
        }
        let mut sum_exp = 0.0f32;
        for k in 0..cf {
            sum_exp += (lf_vec[i][k] - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        total_ce_f += -(lf_vec[i][y] - lse);
    }
    let ce_forget = total_ce_f / (bf as f32);
    let l_unlearn = -ce_forget.min(tau);

    // 2. Retain KL divergence
    let mut total_kl = 0.0f32;
    for i in 0..br {
        // Ref probabilities and log probabilities
        let mut max_rf = lr_rf[i][0];
        let mut max_th = lr_th[i][0];
        for k in 1..cr {
            if lr_rf[i][k] > max_rf {
                max_rf = lr_rf[i][k];
            }
            if lr_th[i][k] > max_th {
                max_th = lr_th[i][k];
            }
        }
        let mut sum_exp_rf = 0.0f32;
        let mut sum_exp_th = 0.0f32;
        for k in 0..cr {
            sum_exp_rf += (lr_rf[i][k] - max_rf).exp();
            sum_exp_th += (lr_th[i][k] - max_th).exp();
        }
        let lse_rf = max_rf + sum_exp_rf.ln();
        let lse_th = max_th + sum_exp_th.ln();

        let mut sample_kl = 0.0f32;
        for k in 0..cr {
            let log_p_rf = lr_rf[i][k] - lse_rf;
            let p_rf = log_p_rf.exp();
            let log_p_th = lr_th[i][k] - lse_th;
            sample_kl += p_rf * (log_p_rf - log_p_th);
        }
        total_kl += sample_kl;
    }
    let kl_retain = total_kl / (br as f32);

    let total = l_unlearn + gamma * kl_retain;
    Tensor::new(total, logits_f_theta.device())
}

/// SISA multi-shard probability aggregation across S models (Bourtoule et al. 2021).
pub fn sisa_ensemble_predict(shard_logits: &Tensor) -> Result<Tensor> {
    let s_vec = shard_logits.to_vec3::<f32>()?;
    let s = s_vec.len();
    let b = s_vec[0].len();
    let c = s_vec[0][0].len();

    let mut mean_probs = vec![vec![0.0f32; c]; b];

    for shard in 0..s {
        for i in 0..b {
            let mut max_val = s_vec[shard][i][0];
            for k in 1..c {
                if s_vec[shard][i][k] > max_val {
                    max_val = s_vec[shard][i][k];
                }
            }
            let mut sum_exp = 0.0f32;
            for k in 0..c {
                sum_exp += (s_vec[shard][i][k] - max_val).exp();
            }
            for k in 0..c {
                let p = (s_vec[shard][i][k] - max_val).exp() / sum_exp;
                mean_probs[i][k] += p / (s as f32);
            }
        }
    }

    Tensor::new(mean_probs, shard_logits.device())
}
