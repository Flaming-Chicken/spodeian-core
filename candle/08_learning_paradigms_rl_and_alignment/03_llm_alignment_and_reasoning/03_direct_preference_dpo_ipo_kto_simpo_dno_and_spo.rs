//! Direct Preference DPO Ipo KTO SimPO DNO and Spo
//! in candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Direct Preference Optimization (DPO) pairwise loss.
pub fn dpo_loss(
    pi_logps_w: &Tensor,
    pi_logps_l: &Tensor,
    ref_logps_w: &Tensor,
    ref_logps_l: &Tensor,
    beta: f32,
) -> Result<Tensor> {
    let pw = pi_logps_w.to_vec1::<f32>()?;
    let pl = pi_logps_l.to_vec1::<f32>()?;
    let rw = ref_logps_w.to_vec1::<f32>()?;
    let rl = ref_logps_l.to_vec1::<f32>()?;

    let b = pw.len();
    let mut total = 0.0f32;

    for i in 0..b {
        let pi_diff = pw[i] - pl[i];
        let ref_diff = rw[i] - rl[i];
        let logits = beta * (pi_diff - ref_diff);
        let sp = if -logits > 30.0 {
            -logits
        } else {
            (1.0f32 + (-logits).exp()).ln()
        };
        total += sp;
    }

    Tensor::new(total / (b as f32), pi_logps_w.device())
}

/// Identity Preference Optimization (IPO) quadratic preference loss.
pub fn ipo_loss(
    pi_logps_w: &Tensor,
    pi_logps_l: &Tensor,
    ref_logps_w: &Tensor,
    ref_logps_l: &Tensor,
    beta: f32,
    tau: f32,
) -> Result<Tensor> {
    let pw = pi_logps_w.to_vec1::<f32>()?;
    let pl = pi_logps_l.to_vec1::<f32>()?;
    let rw = ref_logps_w.to_vec1::<f32>()?;
    let rl = ref_logps_l.to_vec1::<f32>()?;

    let b = pw.len();
    let target = 1.0f32 / (2.0f32 * tau);
    let mut total = 0.0f32;

    for i in 0..b {
        let pi_diff = pw[i] - pl[i];
        let ref_diff = rw[i] - rl[i];
        let logits = beta * (pi_diff - ref_diff);
        let diff = logits - target;
        total += diff * diff;
    }

    Tensor::new(total / (b as f32), pi_logps_w.device())
}

/// Simple Preference Optimization (SimPO) reference-free length-normalized loss.
pub fn simpo_loss(
    pi_logps_w: &Tensor,
    pi_logps_l: &Tensor,
    len_w: &Tensor,
    len_l: &Tensor,
    beta: f32,
    gamma_margin: f32,
) -> Result<Tensor> {
    let pw = pi_logps_w.to_vec1::<f32>()?;
    let pl = pi_logps_l.to_vec1::<f32>()?;
    let lw = len_w.to_vec1::<f32>()?;
    let ll = len_l.to_vec1::<f32>()?;

    let b = pw.len();
    let mut total = 0.0f32;

    for i in 0..b {
        let norm_w = pw[i] / lw[i];
        let norm_l = pl[i] / ll[i];
        let logits = beta * (norm_w - norm_l) - gamma_margin;
        let sp = if -logits > 30.0 {
            -logits
        } else {
            (1.0f32 + (-logits).exp()).ln()
        };
        total += sp;
    }

    Tensor::new(total / (b as f32), pi_logps_w.device())
}

/// Kahneman-Tversky Optimization (KTO) unpaired prospect theory loss.
pub fn kto_loss(
    pi_logps: &Tensor,
    ref_logps: &Tensor,
    desirable_flags: &Tensor,
    beta: f32,
    desirable_weight: f32,
    undesirable_weight: f32,
) -> Result<Tensor> {
    let p = pi_logps.to_vec1::<f32>()?;
    let r = ref_logps.to_vec1::<f32>()?;
    let flags = desirable_flags.to_vec1::<f32>()?;

    let b = p.len();
    let mut total = 0.0f32;

    for i in 0..b {
        let log_ratio = beta * (p[i] - r[i]);
        let sig_neg = 1.0f32 / (1.0f32 + log_ratio.exp());
        let sig_pos = 1.0f32 / (1.0f32 + (-log_ratio).exp());
        let loss = if flags[i] > 0.5 {
            sig_neg * desirable_weight
        } else {
            sig_pos * undesirable_weight
        };
        total += loss;
    }

    Tensor::new(total / (b as f32), pi_logps.device())
}
