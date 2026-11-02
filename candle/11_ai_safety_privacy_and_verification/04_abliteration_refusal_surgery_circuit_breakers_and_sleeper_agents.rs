//! Abliteration Refusal Surgery Circuit Breakers and Sleeper Agents
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Extracts unit refusal direction r_hat = (mu_harmful - mu_harmless) / ||mu_harmful - mu_harmless||_2.
pub fn extract_refusal_direction(act_harmful: &Tensor, act_harmless: &Tensor) -> Result<Tensor> {
    let h_vec = act_harmful.to_vec2::<f32>()?;
    let b_vec = act_harmless.to_vec2::<f32>()?;

    let n_h = h_vec.len();
    let n_b = b_vec.len();
    let d = h_vec[0].len();

    let mut diff = vec![0.0f32; d];
    for j in 0..d {
        let mut sum_h = 0.0f32;
        for i in 0..n_h {
            sum_h += h_vec[i][j];
        }
        let mut sum_b = 0.0f32;
        for i in 0..n_b {
            sum_b += b_vec[i][j];
        }
        diff[j] = (sum_h / n_h as f32) - (sum_b / n_b as f32);
    }

    let mut norm_sq = 0.0f32;
    for j in 0..d {
        norm_sq += diff[j] * diff[j];
    }
    let norm = norm_sq.sqrt() + 1e-8;

    for j in 0..d {
        diff[j] /= norm;
    }

    Tensor::new(diff, act_harmful.device())
}

/// Abliterates refusal mechanism by projecting weight matrix orthogonal to refusal direction:
/// W' = W - r_hat * (r_hat^T W).
pub fn abliterate_weights(weight: &Tensor, refusal_direction: &Tensor) -> Result<Tensor> {
    let w_vec = weight.to_vec2::<f32>()?;
    let r_vec = refusal_direction.to_vec1::<f32>()?;

    let d_in = w_vec.len();
    let d_out = w_vec[0].len();

    // r^T @ W: length d_out
    let mut rt_w = vec![0.0f32; d_out];
    for j in 0..d_out {
        let mut sum = 0.0f32;
        for i in 0..d_in {
            sum += r_vec[i] * w_vec[i][j];
        }
        rt_w[j] = sum;
    }

    // W' = W - r * rt_w
    let mut out = vec![vec![0.0f32; d_out]; d_in];
    for i in 0..d_in {
        for j in 0..d_out {
            out[i][j] = w_vec[i][j] - r_vec[i] * rt_w[j];
        }
    }

    Tensor::new(out, weight.device())
}

/// Contrastive Activation Addition (CAA / RepE): h' = h + alpha * steering_vector.
pub fn contrastive_activation_addition(
    h: &Tensor,
    steering_vector: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    let h_vec = h.to_vec2::<f32>()?;
    let s_vec = steering_vector.to_vec1::<f32>()?;

    let b = h_vec.len();
    let d = h_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            out[i][j] = h_vec[i][j] + alpha * s_vec[j];
        }
    }

    Tensor::new(out, h.device())
}

/// Safety Circuit Breakers loss (Zou et al. 2024 / RepBend).
pub fn circuit_breaker_loss(
    h_harmful: &Tensor,
    h_target_orthogonal: &Tensor,
    h_benign: &Tensor,
    h_ref_benign: &Tensor,
    lambda_cb: f32,
) -> Result<Tensor> {
    let hh_vec = h_harmful.to_vec2::<f32>()?;
    let ht_vec = h_target_orthogonal.to_vec2::<f32>()?;
    let hb_vec = h_benign.to_vec2::<f32>()?;
    let hr_vec = h_ref_benign.to_vec2::<f32>()?;

    let b = hh_vec.len();
    let d = hh_vec[0].len();

    // 1. Breaker loss: mean cosine similarity squared
    let mut total_cos2 = 0.0f32;
    for i in 0..b {
        let mut dot = 0.0f32;
        let mut norm_h2 = 0.0f32;
        let mut norm_t2 = 0.0f32;
        for j in 0..d {
            dot += hh_vec[i][j] * ht_vec[i][j];
            norm_h2 += hh_vec[i][j] * hh_vec[i][j];
            norm_t2 += ht_vec[i][j] * ht_vec[i][j];
        }
        let denom = (norm_h2.sqrt() + 1e-8) * (norm_t2.sqrt() + 1e-8);
        let cos_sim = dot / denom;
        total_cos2 += cos_sim * cos_sim;
    }
    let l_breaker = total_cos2 / (b as f32);

    // 2. Benign retention: MSE
    let mut total_mse = 0.0f32;
    for i in 0..b {
        for j in 0..d {
            let diff = hb_vec[i][j] - hr_vec[i][j];
            total_mse += diff * diff;
        }
    }
    let l_benign = total_mse / ((b * d) as f32);

    let total = l_breaker + lambda_cb * l_benign;
    Tensor::new(total, h_harmful.device())
}

/// Linear probe for auditing backdoor sleeper agents.
pub fn sleeper_agent_probe(h: &Tensor, probe_w: &Tensor, probe_b: &Tensor) -> Result<Tensor> {
    h.matmul(probe_w)?.broadcast_add(probe_b)
}
