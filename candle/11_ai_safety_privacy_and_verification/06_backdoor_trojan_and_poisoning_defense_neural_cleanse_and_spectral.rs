//! Backdoor Trojan and Poisoning Defense Neural Cleanse and Spectral
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Spectral signature outlier scores via representation covariance SVD (Tran et al. 2018).
pub fn spectral_signature_scores(representations: &Tensor) -> Result<Tensor> {
    let h_vec = representations.to_vec2::<f32>()?;
    let n = h_vec.len();
    let d = h_vec[0].len();

    // 1. Compute mean and center representations
    let mut mu = vec![0.0f32; d];
    for i in 0..n {
        for j in 0..d {
            mu[j] += h_vec[i][j];
        }
    }
    for j in 0..d {
        mu[j] /= n as f32;
    }

    let mut centered = vec![vec![0.0f32; d]; n];
    for i in 0..n {
        for j in 0..d {
            centered[i][j] = h_vec[i][j] - mu[j];
        }
    }

    // 2. Sample covariance matrix: Cov = centered^T @ centered / n
    let mut cov = vec![vec![0.0f32; d]; d];
    for j in 0..d {
        for k in 0..d {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += centered[i][j] * centered[i][k];
            }
            cov[j][k] = sum / (n as f32);
        }
    }

    // 3. Power iteration for top singular vector (30 iterations)
    let mut v = vec![1.0f32 / (d as f32).sqrt(); d];
    for _ in 0..30 {
        let mut v_next = vec![0.0f32; d];
        for j in 0..d {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += cov[j][k] * v[k];
            }
            v_next[j] = sum;
        }
        let mut norm_sq = 0.0f32;
        for j in 0..d {
            norm_sq += v_next[j] * v_next[j];
        }
        let norm = norm_sq.sqrt() + 1e-8;
        for j in 0..d {
            v[j] = v_next[j] / norm;
        }
    }

    // 4. Project centered representations onto top singular vector: score = (c . v)^2
    let mut scores = vec![0.0f32; n];
    for i in 0..n {
        let mut dot = 0.0f32;
        for j in 0..d {
            dot += centered[i][j] * v[j];
        }
        scores[i] = dot * dot;
    }

    Tensor::new(scores, representations.device())
}

/// Neural Cleanse MAD outlier detection index for reversed trigger mask L1 norms (Wang et al. 2019).
pub fn neural_cleanse_anomaly_index(l1_norms: &Tensor) -> Result<Tensor> {
    let s = l1_norms.flatten_all()?.to_vec1::<f32>()?;
    let n = s.len();

    let mut sorted_s = s.clone();
    sorted_s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted_s[n / 2];

    let mut abs_dev: Vec<f32> = s.iter().map(|v| (v - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_dev[n / 2] + 1e-8;

    let anomaly: Vec<f32> = s.iter().map(|v| (med - v) / (1.4826 * mad)).collect();
    Tensor::new(anomaly, l1_norms.device())
}

/// STRIP runtime input trojan detection via Shannon entropy (Gao et al. 2019).
pub fn strip_entropy(superposed_logits: &Tensor) -> Result<Tensor> {
    let l_vec = superposed_logits.to_vec2::<f32>()?;
    let k = l_vec.len();
    let c = l_vec[0].len();

    let mut total_entropy = 0.0f32;
    for i in 0..k {
        let mut max_val = l_vec[i][0];
        for col in 1..c {
            if l_vec[i][col] > max_val {
                max_val = l_vec[i][col];
            }
        }
        let mut sum_exp = 0.0f32;
        for col in 0..c {
            sum_exp += (l_vec[i][col] - max_val).exp();
        }

        let mut sample_h = 0.0f32;
        for col in 0..c {
            let p = (l_vec[i][col] - max_val).exp() / sum_exp;
            sample_h += -p * (p + 1e-12).ln();
        }
        total_entropy += sample_h;
    }

    Tensor::new(total_entropy / (k as f32), superposed_logits.device())
}

/// BadNets blending trojan trigger: x' = x * (1 - mask) + pattern * mask.
pub fn trojan_trigger_apply(x: &Tensor, mask: &Tensor, pattern: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let m_vec = mask.to_vec1::<f32>()?;
    let p_vec = pattern.to_vec1::<f32>()?;

    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            out[i][j] = x_vec[i][j] * (1.0 - m_vec[j]) + p_vec[j] * m_vec[j];
        }
    }

    Tensor::new(out, x.device())
}
