//! Survival Analysis and Contextual Bandits in Candle.
//!
//! Features:
//! - Nonparametric Kaplan-Meier survival curve estimation
//! - Cox Proportional Hazards (DeepSurv) partial log-likelihood via risk set aggregation
//! - Linear Upper Confidence Bound (LinUCB) contextual bandit score evaluation
//! - Sherman-Morrison O(d^2) matrix inverse rank-1 updates

use candle_core::{Result, Tensor};

/// Inverts a small square matrix (NxN where N <= 8) using Gauss-Jordan elimination with partial pivoting.
pub fn invert_matrix_cpu(matrix: &[f32], n: usize) -> std::result::Result<Vec<f32>, String> {
    let mut a = vec![vec![0.0f32; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = matrix[i * n + j];
        }
        a[i][n + i] = 1.0f32;
    }

    for i in 0..n {
        // Pivot selection
        let mut pivot_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                pivot_row = k;
            }
        }
        if max_val < 1e-12 {
            return Err("Singular matrix in inversion".to_string());
        }
        a.swap(i, pivot_row);

        let pivot = a[i][i];
        for j in 0..(2 * n) {
            a[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in 0..(2 * n) {
                    a[k][j] -= factor * a[i][j];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i][n + j];
        }
    }
    Ok(inv)
}

/// Evaluates Kaplan-Meier survival estimator S(t).
pub fn kaplan_meier_survival(t_times: &Tensor, e_events: &Tensor) -> Result<(Tensor, Tensor)> {
    let times: Vec<f32> = t_times.flatten_all()?.to_vec1()?;
    let events: Vec<f32> = e_events.flatten_all()?.to_vec1()?;
    let n = times.len();

    // Collect and sort unique event times
    let mut unique_times: Vec<f32> = times
        .iter()
        .zip(events.iter())
        .filter(|&(_, &e)| e == 1.0)
        .map(|(&t, _)| t)
        .collect();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup();

    let mut surv_probs = Vec::with_capacity(unique_times.len());
    let mut current_s = 1.0f32;

    for &t_i in &unique_times {
        let mut n_i = 0.0f32;
        let mut d_i = 0.0f32;
        for i in 0..n {
            if times[i] >= t_i {
                n_i += 1.0;
            }
            if times[i] == t_i && events[i] == 1.0 {
                d_i += 1.0;
            }
        }
        current_s *= 1.0 - (d_i / n_i);
        surv_probs.push(current_s);
    }

    let k = unique_times.len();
    let km_times = Tensor::from_vec(unique_times, (k,), t_times.device())?;
    let km_surv = Tensor::from_vec(surv_probs, (k,), t_times.device())?;
    Ok((km_times, km_surv))
}

/// Evaluates Cox Proportional Hazards negative partial log-likelihood.
pub fn cox_ph_loss(risk_scores: &Tensor, t_times: &Tensor, e_events: &Tensor) -> Result<Tensor> {
    let risk: Vec<f32> = risk_scores.flatten_all()?.to_vec1()?;
    let times: Vec<f32> = t_times.flatten_all()?.to_vec1()?;
    let events: Vec<f32> = e_events.flatten_all()?.to_vec1()?;
    let n = risk.len();

    // Sort indices descending by time
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| times[j].partial_cmp(&times[i]).unwrap());

    // Prefix logcumsumexp over risk set
    let mut log_risk_set = vec![0.0f32; n];
    let mut running_lse = f32::NEG_INFINITY;

    for i in 0..n {
        let r_val = risk[indices[i]];
        if running_lse == f32::NEG_INFINITY {
            running_lse = r_val;
        } else {
            let max_val = running_lse.max(r_val);
            let min_val = running_lse.min(r_val);
            running_lse = max_val + (1.0f32 + (min_val - max_val).exp()).ln();
        }
        log_risk_set[i] = running_lse;
    }

    let mut total_loss = 0.0f32;
    for i in 0..n {
        let idx = indices[i];
        if events[idx] == 1.0 {
            total_loss -= risk[idx] - log_risk_set[i];
        }
    }

    Tensor::new(&[total_loss], risk_scores.device())
}

/// Evaluates LinUCB exploration scores across arms.
pub fn linucb_score_arms(
    x_context: &Tensor,
    a_arms: &Tensor,
    b_arms: &Tensor,
    alpha: f64,
) -> Result<(Tensor, Tensor)> {
    let (num_arms, d, _) = a_arms.dims3()?;
    let x_vec: Vec<f32> = x_context.flatten_all()?.to_vec1()?;
    let a_flat: Vec<f32> = a_arms.flatten_all()?.to_vec1()?;
    let b_flat: Vec<f32> = b_arms.flatten_all()?.to_vec1()?;

    let mut scores = vec![0.0f32; num_arms];
    let mut best_arm = 0usize;
    let mut max_score = f32::NEG_INFINITY;

    for a in 0..num_arms {
        let a_slice = &a_flat[a * d * d..(a + 1) * d * d];
        let a_inv = invert_matrix_cpu(a_slice, d).map_err(candle_core::Error::msg)?;

        let b_slice = &b_flat[a * d..(a + 1) * d];

        // theta_hat = A_inv @ b
        let mut theta_hat = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += a_inv[i * d + j] * b_slice[j];
            }
            theta_hat[i] = sum;
        }

        // mean_reward = x @ theta_hat
        let mut mean_reward = 0.0f32;
        for i in 0..d {
            mean_reward += x_vec[i] * theta_hat[i];
        }

        // x @ A_inv @ x
        let mut a_inv_x = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += a_inv[i * d + j] * x_vec[j];
            }
            a_inv_x[i] = sum;
        }
        let mut var = 0.0f32;
        for i in 0..d {
            var += x_vec[i] * a_inv_x[i];
        }
        let conf_width = (alpha as f32) * var.max(0.0).sqrt();

        let arm_score = mean_reward + conf_width;
        scores[a] = arm_score;
        if arm_score > max_score {
            max_score = arm_score;
            best_arm = a;
        }
    }

    let scores_t = Tensor::from_vec(scores, (num_arms,), x_context.device())?;
    let best_t = Tensor::new(&[best_arm as i64], x_context.device())?;
    Ok((scores_t, best_t))
}

/// Updates matrix inverse in O(d^2) time after rank-1 addition A + x x^T:
/// (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
pub fn sherman_morrison_rank1_update(a_inv: &Tensor, x: &Tensor) -> Result<Tensor> {
    let _d = x.dim(0)?;
    let x_col = x.unsqueeze(1)?; // [D, 1]
    let a_inv_x = a_inv.matmul(&x_col)?; // [D, 1]
    let a_inv_x_t = a_inv_x.t()?; // [1, D]

    let dot = x_col.t()?.matmul(&a_inv_x)?.squeeze(0)?.squeeze(0)?;
    let denom = (dot.to_scalar::<f32>()? + 1.0f32) as f64;

    let outer = a_inv_x.matmul(&a_inv_x_t)?; // [D, D]
    let scaled_outer = outer.affine(1.0 / denom, 0.0)?;
    a_inv - scaled_outer
}
