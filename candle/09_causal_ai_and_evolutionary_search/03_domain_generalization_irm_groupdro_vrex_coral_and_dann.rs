//! Domain Generalization IRM GroupDRO VREx CORAL and DANN
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Invariant Risk Minimization (IRMv1 - Arjovsky et al. 2019):
/// grad = 1/N sum_i [ logits_i * (sigmoid(logits_i) - targets_i) ]
/// penalty = grad^2
pub fn irm_v1_penalty(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let z = logits.to_vec2::<f32>()?;
    let y = targets.to_vec2::<f32>()?;
    let n = z.len();

    let mut grad_sum = 0.0f32;
    for i in 0..n {
        let prob = 1.0f32 / (1.0f32 + (-z[i][0]).exp());
        grad_sum += z[i][0] * (prob - y[i][0]);
    }
    let grad = grad_sum / (n as f32);
    Tensor::from_vec(vec![grad * grad], 1, logits.device())
}

/// Variance Risk Extrapolation (V-REx - Krueger et al. 2021):
/// penalty = 1/E sum_e (L_e - L_bar)^2
pub fn vrex_variance_penalty(env_losses: &Tensor) -> Result<Tensor> {
    let l = env_losses.to_vec1::<f32>()?;
    let e = l.len();
    let mean: f32 = l.iter().sum::<f32>() / (e as f32);
    let var: f32 = l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (e as f32);
    Tensor::from_vec(vec![var], 1, env_losses.device())
}

/// Group Distributionally Robust Optimization (GroupDRO - Sagawa et al. 2020):
/// loss = sum_e q_e * L_e
/// q'_e = q_e * exp(eta * L_e) / sum_k [ q_k * exp(eta * L_k) ]
pub fn group_dro_update(
    env_losses: &Tensor,
    q_weights: &Tensor,
    eta: f32,
) -> Result<(Tensor, Tensor)> {
    let l = env_losses.to_vec1::<f32>()?;
    let q = q_weights.to_vec1::<f32>()?;
    let e = l.len();

    let mut dro_loss = 0.0f32;
    let mut unnorm_q = Vec::with_capacity(e);
    let mut sum_unnorm = 0.0f32;

    for i in 0..e {
        dro_loss += q[i] * l[i];
        let val = q[i] * (eta * l[i]).exp();
        unnorm_q.push(val);
        sum_unnorm += val;
    }

    let q_up: Vec<f32> = unnorm_q.into_iter().map(|v| v / sum_unnorm).collect();
    let loss_t = Tensor::from_vec(vec![dro_loss], 1, env_losses.device())?;
    let q_t = Tensor::from_vec(q_up, e, env_losses.device())?;
    Ok((loss_t, q_t))
}

/// Deep CORAL (Sun & Saenko 2016) second-order covariance alignment loss:
/// loss = 1 / (4 * d^2) * || C_s - C_t ||_F^2
pub fn deep_coral_loss(phi_s: &Tensor, phi_t: &Tensor) -> Result<Tensor> {
    let (ns, d) = phi_s.dims2()?;
    let (nt, _) = phi_t.dims2()?;
    let s = phi_s.to_vec2::<f32>()?;
    let t = phi_t.to_vec2::<f32>()?;

    let mut mean_s = vec![0.0f32; d];
    let mut mean_t = vec![0.0f32; d];
    for i in 0..ns {
        for j in 0..d {
            mean_s[j] += s[i][j];
        }
    }
    for i in 0..nt {
        for j in 0..d {
            mean_t[j] += t[i][j];
        }
    }
    for j in 0..d {
        mean_s[j] /= ns as f32;
        mean_t[j] /= nt as f32;
    }

    let mut cov_s = vec![vec![0.0f32; d]; d];
    let mut cov_t = vec![vec![0.0f32; d]; d];
    for r in 0..d {
        for c in 0..d {
            let mut sum_s = 0.0f32;
            for i in 0..ns {
                sum_s += (s[i][r] - mean_s[r]) * (s[i][c] - mean_s[c]);
            }
            cov_s[r][c] = sum_s / ((ns - 1) as f32);

            let mut sum_t = 0.0f32;
            for i in 0..nt {
                sum_t += (t[i][r] - mean_t[r]) * (t[i][c] - mean_t[c]);
            }
            cov_t[r][c] = sum_t / ((nt - 1) as f32);
        }
    }

    let mut diff_sq = 0.0f32;
    for r in 0..d {
        for c in 0..d {
            let diff = cov_s[r][c] - cov_t[r][c];
            diff_sq += diff * diff;
        }
    }

    let coral = diff_sq / (4.0 * (d * d) as f32);
    Tensor::from_vec(vec![coral], 1, phi_s.device())
}
