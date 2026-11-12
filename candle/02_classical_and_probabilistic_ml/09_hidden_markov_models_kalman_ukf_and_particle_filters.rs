//! Hidden Markov Models, Kalman Filters, UKF, and Particle Filters in Candle.
//!
//! Features:
//! - Linear Kalman Filter with Joseph-stabilized covariance updates
//! - Discrete Hidden Markov Model with log-domain Forward inference and Viterbi decoding
//! - Unscented Kalman Filter (UKF) deterministic sigma-point generation

use candle_core::{Result, Tensor};

/// Inverts a 1x1 or 2x2 matrix analytically.
fn invert_small_matrix(m: &Tensor) -> Result<Tensor> {
    let dims = m.dims();
    if dims == [1, 1] {
        let val = m.flatten_all()?.to_vec1::<f32>()?[0];
        Tensor::new(&[[1.0f32 / val]], m.device())
    } else if dims == [2, 2] {
        let vals: Vec<Vec<f32>> = m.to_vec2()?;
        let a = vals[0][0];
        let b = vals[0][1];
        let c = vals[1][0];
        let d = vals[1][1];
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            candle_core::bail!("Singular matrix in inversion: det = {det}");
        }
        let inv_det = 1.0f32 / det;
        Tensor::new(
            &[[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]],
            m.device(),
        )
    } else {
        candle_core::bail!("Unsupported matrix shape for inversion: {:?}", dims);
    }
}

/// Runs forward Linear Kalman Filter with Joseph-stabilized covariance update:
/// P = (I - KH) P_pred (I - KH)^T + K R K^T
pub fn kalman_filter_forward(
    y: &Tensor,
    f: &Tensor,
    h: &Tensor,
    q: &Tensor,
    r: &Tensor,
    x0: &Tensor,
    p0: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let t_steps = y.dim(0)?;
    let d_state = f.dim(0)?;
    let device = y.device();

    let mut x_hat = if x0.rank() == 1 {
        x0.unsqueeze(1)?
    } else {
        x0.clone()
    };
    let mut p_hat = p0.clone();

    // Eye matrix for Joseph form
    let mut eye_data = vec![0.0f32; d_state * d_state];
    for i in 0..d_state {
        eye_data[i * d_state + i] = 1.0f32;
    }
    let i_mat = Tensor::from_vec(eye_data, (d_state, d_state), device)?;

    let mut x_filt_list = Vec::with_capacity(t_steps);
    let mut p_diag_list = Vec::with_capacity(t_steps);

    for t in 0..t_steps {
        let y_t = y.get(t)?; // [d_y] or [d_y, 1]
        let y_col = if y_t.rank() == 1 {
            y_t.unsqueeze(1)?
        } else {
            y_t
        };

        // Predict
        let x_pred = f.matmul(&x_hat)?;
        let f_t = f.t()?;
        let p_pred = (f.matmul(&p_hat)?.matmul(&f_t)? + q)?;

        // Innovation
        let h_t = h.t()?;
        let h_x_pred = h.matmul(&x_pred)?;
        let innov = (&y_col - &h_x_pred)?;
        let s = (h.matmul(&p_pred)?.matmul(&h_t)? + r)?;

        // Kalman gain: K = P_pred @ H^T @ S^{-1}
        let s_inv = invert_small_matrix(&s)?;
        let k = p_pred.matmul(&h_t)?.matmul(&s_inv)?;

        // Update state
        x_hat = (&x_pred + k.matmul(&innov)?)?;

        // Joseph stabilized covariance update: P = (I - KH) P_pred (I - KH)^T + K R K^T
        let kh = k.matmul(h)?;
        let i_kh = (&i_mat - &kh)?;
        let i_kh_t = i_kh.t()?;
        let k_t = k.t()?;
        let p_part1 = i_kh.matmul(&p_pred)?.matmul(&i_kh_t)?;
        let p_part2 = k.matmul(r)?.matmul(&k_t)?;
        p_hat = (&p_part1 + &p_part2)?;

        // Record filtered state and diagonal variance
        let x_flat = x_hat.squeeze(1)?;
        let p_diag_vals: Vec<f32> = {
            let p_mat: Vec<Vec<f32>> = p_hat.to_vec2()?;
            (0..d_state).map(|i| p_mat[i][i]).collect()
        };
        let p_diag = Tensor::from_vec(p_diag_vals, (d_state,), device)?;

        x_filt_list.push(x_flat);
        p_diag_list.push(p_diag);
    }

    let x_filt = Tensor::stack(&x_filt_list, 0)?;
    let p_filt_diag = Tensor::stack(&p_diag_list, 0)?;

    Ok((x_filt, p_filt_diag))
}

/// Computes log-domain logsumexp over an array of floats.
fn logsumexp(vals: &[f32]) -> f32 {
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max_val.is_infinite() {
        return f32::NEG_INFINITY;
    }
    let sum_exp: f32 = vals.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Evaluates HMM log likelihood using log-domain Forward algorithm.
pub fn hmm_forward_log_prob(
    obs: &Tensor,
    log_pi: &Tensor,
    log_a: &Tensor,
    log_b: &Tensor,
) -> Result<Tensor> {
    let obs_vec: Vec<i64> = obs.to_vec1()?;
    let t_steps = obs_vec.len();
    let s_states = log_pi.dim(0)?;

    let pi_vec: Vec<f32> = log_pi.to_vec1()?;
    let a_mat: Vec<Vec<f32>> = log_a.to_vec2()?;
    let b_mat: Vec<Vec<f32>> = log_b.to_vec2()?;

    let mut alpha = vec![0.0f32; s_states];
    for s in 0..s_states {
        alpha[s] = pi_vec[s] + b_mat[s][obs_vec[0] as usize];
    }

    for t in 1..t_steps {
        let o = obs_vec[t] as usize;
        let mut next_alpha = vec![0.0f32; s_states];
        for s in 0..s_states {
            let mut trans_vals = vec![0.0f32; s_states];
            for prev in 0..s_states {
                trans_vals[prev] = alpha[prev] + a_mat[prev][s];
            }
            next_alpha[s] = logsumexp(&trans_vals) + b_mat[s][o];
        }
        alpha = next_alpha;
    }

    let total_ll = logsumexp(&alpha);
    Tensor::new(&[total_ll], log_pi.device())
}

/// Computes most likely state sequence via Viterbi (max-product) algorithm.
pub fn hmm_viterbi_decode(
    obs: &Tensor,
    log_pi: &Tensor,
    log_a: &Tensor,
    log_b: &Tensor,
) -> Result<Tensor> {
    let obs_vec: Vec<i64> = obs.to_vec1()?;
    let t_steps = obs_vec.len();
    let s_states = log_pi.dim(0)?;

    let pi_vec: Vec<f32> = log_pi.to_vec1()?;
    let a_mat: Vec<Vec<f32>> = log_a.to_vec2()?;
    let b_mat: Vec<Vec<f32>> = log_b.to_vec2()?;

    let mut viterbi_probs = vec![vec![0.0f32; s_states]; t_steps];
    let mut backpointers = vec![vec![0usize; s_states]; t_steps];

    for s in 0..s_states {
        viterbi_probs[0][s] = pi_vec[s] + b_mat[s][obs_vec[0] as usize];
    }

    for t in 1..t_steps {
        let o = obs_vec[t] as usize;
        for s in 0..s_states {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0usize;
            for prev in 0..s_states {
                let score = viterbi_probs[t - 1][prev] + a_mat[prev][s];
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }
            backpointers[t][s] = best_prev;
            viterbi_probs[t][s] = best_score + b_mat[s][o];
        }
    }

    let mut best_last = 0usize;
    let mut max_last_prob = f32::NEG_INFINITY;
    for s in 0..s_states {
        if viterbi_probs[t_steps - 1][s] > max_last_prob {
            max_last_prob = viterbi_probs[t_steps - 1][s];
            best_last = s;
        }
    }

    let mut path = vec![0i64; t_steps];
    path[t_steps - 1] = best_last as i64;
    for t in (0..t_steps - 1).rev() {
        path[t] = backpointers[t + 1][path[t + 1] as usize] as i64;
    }

    Tensor::from_vec(path, (t_steps,), obs.device())
}

/// Generates 2n+1 Merwe scaled unscented sigma points.
pub fn generate_ukf_sigma_points(
    mu: &Tensor,
    sigma: &Tensor,
    alpha: f64,
    _beta: f64,
    kappa: f64,
) -> Result<Tensor> {
    let n_dim = mu.dim(0)?;
    let lambda_val = (alpha * alpha) * ((n_dim as f64) + kappa) - (n_dim as f64);
    let gamma = ((n_dim as f64) + lambda_val).sqrt() as f32;

    let mu_vec: Vec<f32> = mu.to_vec1()?;
    let sigma_mat: Vec<Vec<f32>> = sigma.to_vec2()?;

    // Cholesky decomposition of Sigma (lower triangular L)
    let mut l = vec![vec![0.0f32; n_dim]; n_dim];
    for i in 0..n_dim {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = sigma_mat[i][i] - sum;
                if diag <= 0.0 {
                    candle_core::bail!("Covariance matrix is not positive definite: diag = {diag}");
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (sigma_mat[i][j] - sum) / l[j][j];
            }
        }
    }

    let num_points = 2 * n_dim + 1;
    let mut pts = Vec::with_capacity(num_points);

    // 0: mu
    pts.push(mu_vec.clone());

    // 1..=n: mu + gamma * L[:, i]
    for i in 0..n_dim {
        let mut pt = vec![0.0f32; n_dim];
        for d in 0..n_dim {
            pt[d] = mu_vec[d] + gamma * l[d][i];
        }
        pts.push(pt);
    }

    // n+1..=2n: mu - gamma * L[:, i]
    for i in 0..n_dim {
        let mut pt = vec![0.0f32; n_dim];
        for d in 0..n_dim {
            pt[d] = mu_vec[d] - gamma * l[d][i];
        }
        pts.push(pt);
    }

    let flat_pts: Vec<f32> = pts.into_iter().flatten().collect();
    Tensor::from_vec(flat_pts, (num_points, n_dim), mu.device())
}
