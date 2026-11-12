//! k-Nearest Neighbors, Prototype Learning, and Locally Weighted Regression in Native Rust Candle.
//!
//! Provides:
//! - Pairwise squared Euclidean distance calculation with zero-clamping
//! - Distance-weighted k-NN continuous regression
//! - Nadaraya-Watson Gaussian kernel regression

use candle_core::{Result, Tensor};

/// Computes pairwise squared Euclidean distances: D_ij = ||x1_i - x2_j||^2.
pub fn pairwise_sq_dist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let (n1, _d) = x1.dims2()?;
    let (n2, _) = x2.dims2()?;

    let x1_sq = x1.sqr()?.sum_keepdim(1)?; // [n1, 1]
    let x2_sq = x2.sqr()?.sum_keepdim(1)?.t()?; // [1, n2]

    let dot = x1.matmul(&x2.t()?)?; // [n1, n2]
    let dist_sq =
        ((x1_sq.broadcast_as((n1, n2))? + x2_sq.broadcast_as((n1, n2))?)? - (dot * 2.0)?)?;
    dist_sq.clamp(0.0f32, f32::MAX)
}

/// Distance-weighted k-NN continuous regression prediction for queries x_test against (x_train, y_train).
pub fn knn_regress(
    x_train: &Tensor,
    y_train: &Tensor,
    x_test: &Tensor,
    k: usize,
    eps: f32,
) -> Result<Tensor> {
    let (n_train, _) = x_train.dims2()?;
    let (n_test, _) = x_test.dims2()?;

    let dist_sq = pairwise_sq_dist(x_train, x_test)?; // [n_train, n_test]
    let dists = dist_sq.sqrt()?;

    let dists_vec = dists.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = y_train.flatten_all()?.to_vec1::<f32>()?;

    let mut preds = Vec::with_capacity(n_test);

    for j in 0..n_test {
        // Collect (distance, index) pairs for query j
        let mut d_pairs: Vec<(f32, usize)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let d = dists_vec[i * n_test + j];
            d_pairs.push((d, i));
        }

        // Sort by ascending distance
        d_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Top-k weighted average
        let mut sum_weights = 0.0f32;
        let mut sum_weighted_y = 0.0f32;

        for &(d, idx) in d_pairs.iter().take(k) {
            let w = 1.0f32 / (d + eps);
            sum_weights += w;
            sum_weighted_y += w * y_vec[idx];
        }

        let pred = if sum_weights > 1e-8 {
            sum_weighted_y / sum_weights
        } else {
            0.0f32
        };
        preds.push(pred);
    }

    Tensor::from_vec(preds, (n_test,), x_test.device())
}

/// Nadaraya-Watson kernel regression: y(x) = sum_i K_h(x, x_i) y_i / sum_i K_h(x, x_i).
pub fn nadaraya_watson_regress(
    x_train: &Tensor,
    y_train: &Tensor,
    x_test: &Tensor,
    bandwidth: f64,
) -> Result<Tensor> {
    let (n_train, _) = x_train.dims2()?;
    let (n_test, _) = x_test.dims2()?;

    let dist_sq = pairwise_sq_dist(x_train, x_test)?; // [n_train, n_test]
    let factor = -0.5 / (bandwidth * bandwidth);
    let weights = (dist_sq * factor)?.exp()?; // [n_train, n_test]

    let weights_vec = weights.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = y_train.flatten_all()?.to_vec1::<f32>()?;

    let mut preds = Vec::with_capacity(n_test);

    for j in 0..n_test {
        let mut sum_w = 0.0f32;
        let mut sum_wy = 0.0f32;
        for i in 0..n_train {
            let w = weights_vec[i * n_test + j];
            sum_w += w;
            sum_wy += w * y_vec[i];
        }
        let pred = if sum_w > 1e-8 { sum_wy / sum_w } else { 0.0f32 };
        preds.push(pred);
    }

    Tensor::from_vec(preds, (n_test,), x_test.device())
}
