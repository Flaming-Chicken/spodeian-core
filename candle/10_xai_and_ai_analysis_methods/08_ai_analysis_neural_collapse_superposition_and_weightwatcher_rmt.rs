//! AI Analysis Neural Collapse Superposition and WeightWatcher RMT
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Neural Collapse NC1 (within-class variability) and NC2 (Simplex ETF) metrics (Papyan et al., 2020).
pub fn neural_collapse_metrics(
    features: &Tensor,
    labels: &Tensor,
    num_classes: usize,
) -> Result<(Tensor, Tensor)> {
    let n = features.dim(0)?;
    let d = features.dim(1)?;
    let feats = features.to_vec2::<f32>()?;
    let lbls = labels.to_vec1::<f32>()?;
    let c = num_classes;

    let mut class_means = vec![vec![0.0f32; d]; c];
    let mut class_counts = vec![0.0f32; c];
    for i in 0..n {
        let l = lbls[i] as usize;
        for dim in 0..d {
            class_means[l][dim] += feats[i][dim];
        }
        class_counts[l] += 1.0;
    }
    for k in 0..c {
        if class_counts[k] > 0.0 {
            for dim in 0..d {
                class_means[k][dim] /= class_counts[k];
            }
        }
    }

    // NC1
    let mut nc1_sum = 0.0f32;
    for i in 0..n {
        let l = lbls[i] as usize;
        for dim in 0..d {
            let diff = feats[i][dim] - class_means[l][dim];
            nc1_sum += diff * diff;
        }
    }
    let nc1_var = nc1_sum / (n as f32);

    // NC2 Simplex ETF
    let mut global_mean = vec![0.0f32; d];
    for k in 0..c {
        for dim in 0..d {
            global_mean[dim] += class_means[k][dim];
        }
    }
    for dim in 0..d {
        global_mean[dim] /= c as f32;
    }

    let mut norm_means = vec![vec![0.0f32; d]; c];
    for k in 0..c {
        let mut norm_sq = 0.0f32;
        for dim in 0..d {
            let centered = class_means[k][dim] - global_mean[dim];
            norm_means[k][dim] = centered;
            norm_sq += centered * centered;
        }
        let norm = (norm_sq + 1e-8).sqrt();
        for dim in 0..d {
            norm_means[k][dim] /= norm;
        }
    }

    let ideal_cos = -1.0f32 / ((c - 1) as f32);
    let mut nc2_err_sum = 0.0f32;
    for i in 0..c {
        for j in 0..c {
            if i != j {
                let mut cos = 0.0f32;
                for dim in 0..d {
                    cos += norm_means[i][dim] * norm_means[j][dim];
                }
                let diff = cos - ideal_cos;
                nc2_err_sum += diff * diff;
            }
        }
    }
    let nc2_etf_err = nc2_err_sum / ((c * (c - 1)) as f32);

    Ok((
        Tensor::new(nc1_var, features.device())?,
        Tensor::new(nc2_etf_err, features.device())?,
    ))
}

/// Toy Model of Superposition (Elhage et al., 2022) feature interference energy.
pub fn superposition_interference(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let gram = w.t()?.matmul(w)?;
    let m = gram.dim(0)?;
    let gram_vec = gram.to_vec2::<f32>()?;

    let mut interf = 0.0f32;
    for i in 0..m {
        for j in 0..m {
            if i != j {
                interf += gram_vec[i][j] * gram_vec[i][j];
            }
        }
    }

    Ok((gram, Tensor::new(interf, w.device())?))
}

/// WeightWatcher / Heavy-Tailed RMT power-law tail exponent via Hill estimator (Martin & Mahoney, 2021).
pub fn weightwatcher_alpha_hill(eigenvalues: &Tensor, lambda_min: f32) -> Result<Tensor> {
    let evals = eigenvalues.to_vec1::<f32>()?;
    let mut k_tail = 0.0f32;
    let mut sum_log = 0.0f32;

    for &val in &evals {
        if val >= lambda_min {
            k_tail += 1.0;
            sum_log += (val / lambda_min).ln();
        }
    }

    let alpha = 1.0 + k_tail / (sum_log + 1e-8);
    Tensor::new(alpha, eigenvalues.device())
}
