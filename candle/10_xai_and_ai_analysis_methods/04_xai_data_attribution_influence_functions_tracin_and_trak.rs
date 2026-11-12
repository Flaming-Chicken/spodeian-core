//! XAI Data Attribution Influence Functions TracIn and TRAK
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    let mut m = vec![vec![0.0f32; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = a[i][j];
        }
        m[i][n] = b[i];
    }
    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if m[k][i].abs() > m[max_row][i].abs() {
                max_row = k;
            }
        }
        m.swap(i, max_row);
        let pivot = m[i][i];
        if pivot.abs() > 1e-12 {
            for j in i..=n {
                m[i][j] /= pivot;
            }
            for k in 0..n {
                if k != i {
                    let factor = m[k][i];
                    for j in i..=n {
                        m[k][j] -= factor * m[i][j];
                    }
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}

/// Influence function computation via damped inverse Hessian-vector product (Koh & Liang, 2017).
pub fn influence_score(
    hessian: &Tensor,
    grad_test: &Tensor,
    grad_train: &Tensor,
    damping: f32,
) -> Result<(Tensor, Tensor)> {
    let d = hessian.dim(0)?;
    let h_mat = hessian.to_vec2::<f32>()?;
    let g_test = grad_test.to_vec1::<f32>()?;
    let g_train = grad_train.to_vec1::<f32>()?;

    let mut h_damped = h_mat;
    for i in 0..d {
        h_damped[i][i] += damping;
    }

    let ihvp_vec = solve_linear_system(&h_damped, &g_test);
    let mut inf = 0.0f32;
    for i in 0..d {
        inf -= g_train[i] * ihvp_vec[i];
    }

    let dev = hessian.device();
    let ihvp_t = Tensor::from_vec(ihvp_vec, (d,), dev)?;
    let inf_t = Tensor::new(inf, dev)?;

    Ok((ihvp_t, inf_t))
}

/// TracIn: Tracing gradient descent checkpoint dot-products (Pruthi et al., 2020).
pub fn tracin_trajectory_attribution(
    train_grads: &Tensor,
    test_grads: &Tensor,
    learning_rates: &Tensor,
) -> Result<Tensor> {
    let t_g = train_grads.to_vec2::<f32>()?;
    let q_g = test_grads.to_vec2::<f32>()?;
    let lrs = learning_rates.to_vec1::<f32>()?;
    let t = lrs.len();

    let mut score = 0.0f32;
    for step in 0..t {
        let mut dot = 0.0f32;
        for d in 0..t_g[step].len() {
            dot += t_g[step][d] * q_g[step][d];
        }
        score += lrs[step] * dot;
    }

    Tensor::new(score, train_grads.device())
}

/// TRAK: Tracing with random JL projections at scale (Park et al., 2023).
pub fn trak_projected_attribution(
    train_grads: &Tensor,
    test_grad: &Tensor,
    proj_matrix: &Tensor,
) -> Result<Tensor> {
    let phi_train = train_grads.matmul(proj_matrix)?;
    let phi_test = test_grad.unsqueeze(0)?.matmul(proj_matrix)?.squeeze(0)?;

    let p_train = phi_train.to_vec2::<f32>()?;
    let p_test = phi_test.to_vec1::<f32>()?;

    let mut norm_test_sq = 0.0f32;
    for v in &p_test {
        norm_test_sq += v * v;
    }
    let norm_test = (norm_test_sq + 1e-8).sqrt();

    let mut scores = Vec::with_capacity(p_train.len());
    for row in &p_train {
        let mut dot = 0.0f32;
        let mut norm_tr_sq = 0.0f32;
        for k in 0..p_test.len() {
            dot += row[k] * p_test[k];
            norm_tr_sq += row[k] * row[k];
        }
        let norm_tr = (norm_tr_sq + 1e-8).sqrt();
        scores.push(dot / (norm_tr * norm_test));
    }

    Tensor::from_vec(scores, (p_train.len(),), train_grads.device())
}
