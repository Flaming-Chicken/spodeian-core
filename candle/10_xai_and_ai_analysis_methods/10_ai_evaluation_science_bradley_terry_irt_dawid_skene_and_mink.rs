//! AI Evaluation Science Bradley Terry IRT Dawid Skene and Min-K%
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Bradley-Terry-Luce pairwise win probability matrix P_{ij} = sigma(s_i - s_j) (Bradley & Terry, 1952).
pub fn btl_win_probability(skills: &Tensor) -> Result<Tensor> {
    let s = skills.to_vec1::<f32>()?;
    let m = s.len();
    let mut probs = vec![vec![0.0f32; m]; m];

    for i in 0..m {
        for j in 0..m {
            let diff = s[i] - s[j];
            probs[i][j] = 1.0f32 / (1.0f32 + (-diff).exp());
        }
    }
    Tensor::new(probs, skills.device())
}

/// Two-Parameter Logistic Item Response Theory (2PL IRT) P(Y_{mi}=1) = sigma(a_i * (theta_m - b_i)) (Lord, 1980).
pub fn irt_2pl_probability(
    abilities: &Tensor,
    difficulties: &Tensor,
    discriminations: &Tensor,
) -> Result<Tensor> {
    let theta = abilities.to_vec1::<f32>()?;
    let b = difficulties.to_vec1::<f32>()?;
    let a = discriminations.to_vec1::<f32>()?;

    let m = theta.len();
    let items = b.len();
    let mut probs = vec![vec![0.0f32; items]; m];

    for i in 0..m {
        for j in 0..items {
            let diff = theta[i] - b[j];
            let z = a[j] * diff;
            probs[i][j] = 1.0f32 / (1.0f32 + (-z).exp());
        }
    }
    Tensor::new(probs, abilities.device())
}

/// Dawid-Skene annotator consensus expectation step (Dawid & Skene, 1979).
pub fn dawid_skene_e_step(
    annotations: &Tensor,
    class_priors: &Tensor,
    confusion_matrices: &Tensor,
) -> Result<Tensor> {
    let n = annotations.dim(0)?;
    let j_count = annotations.dim(1)?;
    let k_count = class_priors.dim(0)?;

    let annot = annotations.to_vec2::<f32>()?;
    let priors = class_priors.to_vec1::<f32>()?;
    let conf = confusion_matrices.to_vec3::<f32>()?;

    let mut post = vec![vec![0.0f32; k_count]; n];

    for i in 0..n {
        let mut log_p = vec![0.0f32; k_count];
        let mut max_lp = f32::NEG_INFINITY;
        for k in 0..k_count {
            let mut lp = (priors[k] + 1e-8).ln();
            for j in 0..j_count {
                let obs = annot[i][j] as usize;
                lp += (conf[j][k][obs] + 1e-8).ln();
            }
            log_p[k] = lp;
            if lp > max_lp {
                max_lp = lp;
            }
        }

        let mut sum_exp = 0.0f32;
        for k in 0..k_count {
            let exp_val = (log_p[k] - max_lp).exp();
            log_p[k] = exp_val;
            sum_exp += exp_val;
        }

        for k in 0..k_count {
            post[i][k] = log_p[k] / sum_exp;
        }
    }

    Tensor::new(post, annotations.device())
}

/// Min-K% Prob pretraining data / memorization detector (Shi et al., 2023).
pub fn min_k_percent_prob(token_log_probs: &Tensor, k_fraction: f32) -> Result<Tensor> {
    let mut lp = token_log_probs.to_vec1::<f32>()?;
    lp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let t = lp.len();
    let k_count = (1.max(((k_fraction * (t as f32)).round()) as usize)).min(t);

    let sum: f32 = lp.iter().take(k_count).sum();
    let mean = sum / (k_count as f32);

    Tensor::new(mean, token_log_probs.device())
}
