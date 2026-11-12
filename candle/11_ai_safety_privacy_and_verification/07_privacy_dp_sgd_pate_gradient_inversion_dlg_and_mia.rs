//! Privacy DP SGD PATE Gradient Inversion DLG and MIA
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// DP-SGD per-sample gradient clipping and Gaussian noise addition (Abadi et al. 2016).
///
/// \tilde{g} = ( \sum_{i=1}^B g_i \min(1, C / ||g_i||_2) + \sigma C \xi ) / B
pub fn dp_sgd_clip_and_noise(
    per_sample_grads: &Tensor,
    clip_norm: f32,
    noise_vec: &Tensor,
    noise_multiplier: f32,
) -> Result<Tensor> {
    let grads = per_sample_grads.to_vec2::<f32>()?;
    let noise = noise_vec.to_vec1::<f32>()?;
    let b = grads.len();
    let d = grads[0].len();

    let mut clipped_sum = vec![0.0f32; d];
    for i in 0..b {
        let mut norm_sq = 0.0f32;
        for j in 0..d {
            norm_sq += grads[i][j] * grads[i][j];
        }
        let norm = norm_sq.sqrt() + 1e-8f32;
        let scale = (clip_norm / norm).min(1.0f32);
        for j in 0..d {
            clipped_sum[j] += grads[i][j] * scale;
        }
    }

    let mut out = vec![0.0f32; d];
    for j in 0..d {
        let noisy = clipped_sum[j] + noise_multiplier * clip_norm * noise[j];
        out[j] = noisy / (b as f32);
    }

    Tensor::new(out, per_sample_grads.device())
}

/// PATE noisy aggregation with Laplace mechanism (Papernot et al. 2017).
///
/// \tilde{V} = V + \text{Laplace}(0, b)
pub fn pate_noisy_aggregation(
    teacher_votes: &Tensor,
    laplace_noise: &Tensor,
    laplace_scale: f32,
) -> Result<Tensor> {
    let scaled_noise = (laplace_noise * (laplace_scale as f64))?;
    teacher_votes.add(&scaled_noise)
}

/// Deep Leakage from Gradients (DLG) matching loss (Zhu et al. 2019): ||g_dummy - g_target||_2^2.
pub fn dlg_gradient_inversion_loss(dummy_grad: &Tensor, target_grad: &Tensor) -> Result<Tensor> {
    let diff = (dummy_grad - target_grad)?;
    diff.sqr()?.sum_all()
}

/// Likelihood-Ratio Membership Inference Attack (LiRA — Carlini et al. 2022).
///
/// \Lambda = (\mu_{\text{out}} - \mathcal{L}_{\text{target}}) / (\sigma_{\text{out}} + \epsilon)
pub fn lira_membership_score(target_loss: &Tensor, shadow_out_losses: &Tensor) -> Result<Tensor> {
    let tl = target_loss.to_vec1::<f32>()?;
    let sol = shadow_out_losses.to_vec2::<f32>()?;
    let b = sol.len();
    let k = sol[0].len();

    let mut scores = vec![0.0f32; b];
    for i in 0..b {
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += sol[i][j];
        }
        let mu = sum / (k as f32);

        let mut var_sum = 0.0f32;
        for j in 0..k {
            let diff = sol[i][j] - mu;
            var_sum += diff * diff;
        }
        let var = var_sum / (k as f32);
        let sigma = (var + 1e-8f32).sqrt();

        scores[i] = (mu - tl[i]) / sigma;
    }

    Tensor::new(scores, target_loss.device())
}
