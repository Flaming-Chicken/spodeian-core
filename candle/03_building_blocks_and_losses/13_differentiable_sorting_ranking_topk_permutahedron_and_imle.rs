//! Differentiable Sorting, Ranking, Top-k, Permutahedron, and I-MLE (C03 Topic 13).
//!
//! Native Rust Candle implementation of:
//! 1. Linear Anchor.
//! 2. SoftSort continuous relaxation of permutation matrices.
//! 3. NeuralSort unimodal row-stochastic permutation matrix.
//! 4. Differentiable Top-k operator via bisection root-finding.
//! 5. Differentiable Soft Histogram.
//! 6. Differentiable Soft Quantile / Median estimator.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// SoftSort: continuous permutation matrix and sorted values approximation.
pub fn soft_sort_forward(scores: &Tensor, tau: f32) -> Result<(Tensor, Tensor)> {
    let s_vec = scores.to_vec2::<f32>()?;
    let b = s_vec.len();
    let k = s_vec[0].len();

    let mut p_soft_out = vec![vec![vec![0.0f32; k]; k]; b];
    let mut sorted_approx_out = vec![vec![0.0f32; k]; b];

    for b_idx in 0..b {
        let mut s_sorted = s_vec[b_idx].clone();
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..k {
            let s_sort_i = s_sorted[i];
            let mut row = vec![0.0f32; k];
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..k {
                let dist = (s_sort_i - s_vec[b_idx][j]).abs();
                let logit = -dist / tau;
                row[j] = logit;
                if logit > max_val {
                    max_val = logit;
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..k {
                let val = (row[j] - max_val).exp();
                row[j] = val;
                sum_exp += val;
            }
            for j in 0..k {
                row[j] /= sum_exp;
                p_soft_out[b_idx][i][j] = row[j];
            }
        }

        // Reconstructed sorted values: P_soft @ scores
        for i in 0..k {
            let mut val_acc = 0.0f32;
            for j in 0..k {
                val_acc += p_soft_out[b_idx][i][j] * s_vec[b_idx][j];
            }
            sorted_approx_out[b_idx][i] = val_acc;
        }
    }

    let p_tensor = Tensor::new(p_soft_out, scores.device())?;
    let sorted_tensor = Tensor::new(sorted_approx_out, scores.device())?;
    Ok((p_tensor, sorted_tensor))
}

/// NeuralSort: unimodal row-stochastic permutation matrix.
pub fn neural_sort_forward(scores: &Tensor, tau: f32) -> Result<Tensor> {
    let s_vec = scores.to_vec2::<f32>()?;
    let b = s_vec.len();
    let k = s_vec[0].len();

    let mut p_neural_out = vec![vec![vec![0.0f32; k]; k]; b];

    for b_idx in 0..b {
        let s_slice = &s_vec[b_idx];
        let mut l1_sum = vec![0.0f32; k];
        for i in 0..k {
            let mut sum_diff = 0.0f32;
            for j in 0..k {
                sum_diff += (s_slice[i] - s_slice[j]).abs();
            }
            l1_sum[i] = sum_diff;
        }

        for j in 0..k {
            let weight_j = (k + 1) as f32 - 2.0f32 * ((j + 1) as f32);
            let mut row = vec![0.0f32; k];
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..k {
                let cost_ji = (weight_j * s_slice[i] - l1_sum[i]) / tau;
                row[i] = cost_ji;
                if cost_ji > max_val {
                    max_val = cost_ji;
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..k {
                let val = (row[i] - max_val).exp();
                row[i] = val;
                sum_exp += val;
            }
            for i in 0..k {
                p_neural_out[b_idx][j][i] = row[i] / sum_exp;
            }
        }
    }

    Tensor::new(p_neural_out, scores.device())
}

/// Differentiable Top-k via bisection root-finding.
pub fn diff_topk_forward(
    scores: &Tensor,
    k_elements: usize,
    tau: f32,
    num_bisection: usize,
) -> Result<Tensor> {
    let s_vec = scores.to_vec2::<f32>()?;
    let b = s_vec.len();
    let n = s_vec[0].len();

    let mut w_out = vec![vec![0.0f32; n]; b];

    for b_idx in 0..b {
        let s_slice = &s_vec[b_idx];
        let min_s = s_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max_s = s_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut lo = min_s - 5.0f32 * tau;
        let mut hi = max_s + 5.0f32 * tau;

        for _ in 0..num_bisection {
            let mid = 0.5f32 * (lo + hi);
            let mut sum_w = 0.0f32;
            for i in 0..n {
                let sig = 1.0f32 / (1.0f32 + (-(s_slice[i] - mid) / tau).exp());
                sum_w += sig;
            }
            if sum_w > (k_elements as f32) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let mid = 0.5f32 * (lo + hi);
        for i in 0..n {
            w_out[b_idx][i] = 1.0f32 / (1.0f32 + (-(s_slice[i] - mid) / tau).exp());
        }
    }

    Tensor::new(w_out, scores.device())
}

/// Differentiable Soft Histogram.
pub fn soft_histogram_forward(
    x: &Tensor,
    bin_centers: &Tensor,
    bin_width: f32,
    tau: f32,
) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let c_vec = bin_centers.to_vec1::<f32>()?;
    let b = x_vec.len();
    let n = x_vec[0].len();
    let m = c_vec.len();

    let mut hist_out = vec![vec![0.0f32; m]; b];

    for b_idx in 0..b {
        let mut sum_hist = 0.0f32;
        for j in 0..m {
            let c = c_vec[j];
            let lo = c - 0.5f32 * bin_width;
            let hi = c + 0.5f32 * bin_width;

            let mut bin_count = 0.0f32;
            for i in 0..n {
                let val = x_vec[b_idx][i];
                let sig_lo = 1.0f32 / (1.0f32 + (-(val - lo) / tau).exp());
                let sig_hi = 1.0f32 / (1.0f32 + (-(val - hi) / tau).exp());
                bin_count += sig_lo - sig_hi;
            }
            hist_out[b_idx][j] = bin_count;
            sum_hist += bin_count;
        }

        let denom = sum_hist + 1e-12f32;
        for j in 0..m {
            hist_out[b_idx][j] /= denom;
        }
    }

    Tensor::new(hist_out, x.device())
}

/// Differentiable Soft Quantile.
pub fn soft_quantile_forward(scores: &Tensor, quantile: f32, tau: f32) -> Result<Tensor> {
    let s_vec = scores.to_vec2::<f32>()?;
    let b = s_vec.len();
    let n = s_vec[0].len();

    let target_rank = quantile * ((n - 1) as f32) + 1.0f32;
    let mut out = vec![0.0f32; b];

    for b_idx in 0..b {
        let s_slice = &s_vec[b_idx];
        let mut ranks = vec![0.5f32; n];
        for i in 0..n {
            let mut r = 0.5f32;
            for j in 0..n {
                let diff = (s_slice[i] - s_slice[j]) / tau;
                r += 1.0f32 / (1.0f32 + (-diff).exp());
            }
            ranks[i] = r;
        }

        let mut weights = vec![0.0f32; n];
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..n {
            let dist_sq = (ranks[i] - target_rank) * (ranks[i] - target_rank);
            let logit = -dist_sq / (2.0f32 * tau);
            weights[i] = logit;
            if logit > max_val {
                max_val = logit;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..n {
            let val = (weights[i] - max_val).exp();
            weights[i] = val;
            sum_exp += val;
        }

        let mut q_val = 0.0f32;
        for i in 0..n {
            q_val += (weights[i] / sum_exp) * s_slice[i];
        }
        out[b_idx] = q_val;
    }

    Tensor::new(out, scores.device())
}
