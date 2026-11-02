//! Robotics Embodied Ai Diffusion Policy and Act
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Action Chunking with Ensembling (ACT) temporal weighting.
pub fn temporal_ensemble_actions(action_preds: &Tensor, decay: f64) -> Result<Tensor> {
    let k = action_preds.dim(1)?;
    let mut weights = Vec::with_capacity(k);
    let mut sum_w = 0.0f32;
    for i in 0..k {
        let w = (-decay * (i as f64)).exp() as f32;
        weights.push(w);
        sum_w += w;
    }
    for w in weights.iter_mut() {
        *w /= sum_w;
    }

    let w_tensor = Tensor::new(weights, action_preds.device())?.reshape((1, k, 1))?;
    action_preds.broadcast_mul(&w_tensor)?.sum(1)
}

/// Single reverse DDPM denoising step for continuous action diffusion.
pub fn diffusion_policy_ddpm_step(
    action: &Tensor,
    noise_pred: &Tensor,
    noise_z: &Tensor,
    alpha_t: f64,
    alpha_bar_t: f64,
    sigma_t: f64,
) -> Result<Tensor> {
    let coeff = (1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt();
    let mu = ((action - (noise_pred * coeff)?)? * (1.0 / alpha_t.sqrt()))?;
    mu + (noise_z * sigma_t)?
}

/// ACT CVAE Gaussian latent regularization loss.
pub fn act_cvae_kl_loss(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let one_plus_logvar = (1.0 + logvar)?;
    let mu_sq = mu.sqr()?;
    let exp_logvar = logvar.exp()?;
    let term = ((one_plus_logvar - mu_sq)? - exp_logvar)?;
    (term * (-0.5))?.sum(1)?.mean_all()
}

/// Clamps continuous actions within physical safety envelope.
pub fn robot_action_safety_clamp(
    actions: &Tensor,
    a_min: &Tensor,
    a_max: &Tensor,
) -> Result<Tensor> {
    let (b, h, d) = actions.dims3()?;
    let min_exp = a_min.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, d))?;
    let max_exp = a_max.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((b, h, d))?;
    actions.maximum(&min_exp)?.minimum(&max_exp)
}
