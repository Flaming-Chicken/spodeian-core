//! Semi Supervised Fixmatch Mean Teacher and Nnpu Learning
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Mean Teacher consistency MSE loss between student and teacher softmax probabilities.
pub fn mean_teacher_consistency_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
) -> Result<Tensor> {
    let p_s = candle_nn::ops::softmax(student_logits, 1)?;
    let p_t = candle_nn::ops::softmax(teacher_logits, 1)?;
    let loss = (p_s - p_t)?.sqr()?.mean_all()?;
    loss.reshape(&[1])
}

/// FixMatch confidence-thresholded cross-entropy consistency loss.
pub fn fixmatch_thresholded_loss(
    weak_logits: &Tensor,
    strong_logits: &Tensor,
    tau: f32,
) -> Result<Tensor> {
    let q = candle_nn::ops::softmax(weak_logits, 1)?;
    let q_vec = q.to_vec2::<f32>()?;
    let strong_log_p = candle_nn::ops::log_softmax(strong_logits, 1)?.to_vec2::<f32>()?;

    let b = q_vec.len();
    let k = q_vec[0].len();
    let mut loss_sum = 0.0f32;

    for i in 0..b {
        let mut max_p = -1.0f32;
        let mut argmax = 0;
        for j in 0..k {
            if q_vec[i][j] > max_p {
                max_p = q_vec[i][j];
                argmax = j;
            }
        }
        if max_p >= tau {
            loss_sum += -strong_log_p[i][argmax];
        }
    }

    Tensor::new(&[loss_sum / (b as f32)], weak_logits.device())
}

/// MixMatch temperature sharpening: p_k^(1/T) / sum_j p_j^(1/T).
pub fn mixmatch_sharpen(probs: &Tensor, temperature: f32) -> Result<Tensor> {
    let probs_vec = probs.to_vec2::<f32>()?;
    let b = probs_vec.len();
    let k = probs_vec[0].len();
    let inv_t = 1.0f32 / temperature;

    let mut out = Vec::with_capacity(b * k);
    for i in 0..b {
        let mut sum_sharp = 0.0f32;
        let mut sharp_row = vec![0.0f32; k];
        for j in 0..k {
            sharp_row[j] = probs_vec[i][j].powf(inv_t);
            sum_sharp += sharp_row[j];
        }
        for j in 0..k {
            out.push(sharp_row[j] / sum_sharp);
        }
    }

    Tensor::from_vec(out, (b, k), probs.device())
}

/// Non-Negative Positive-Unlabeled (nnPU) risk estimator.
pub fn nnpu_risk(g_pos: &Tensor, g_unlabeled: &Tensor, pi_p: f32) -> Result<Tensor> {
    let g_pos_vec = g_pos.to_vec1::<f32>()?;
    let g_u_vec = g_unlabeled.to_vec1::<f32>()?;

    // softplus(v) = ln(1 + exp(v))
    let sp = |v: f32| (1.0f32 + v.exp()).ln();

    let mut sum_p_pos = 0.0f32;
    let mut sum_p_neg = 0.0f32;
    for &v in &g_pos_vec {
        sum_p_pos += sp(-v);
        sum_p_neg += sp(v);
    }
    let ell_p_pos = sum_p_pos / (g_pos_vec.len() as f32);
    let ell_p_neg = sum_p_neg / (g_pos_vec.len() as f32);

    let mut sum_u_neg = 0.0f32;
    for &v in &g_u_vec {
        sum_u_neg += sp(v);
    }
    let ell_u_neg = sum_u_neg / (g_u_vec.len() as f32);

    let neg_risk = ell_u_neg - pi_p * ell_p_neg;
    let risk = pi_p * ell_p_pos + neg_risk.max(0.0f32);

    Tensor::new(&[risk], g_pos.device())
}
