//! XAI Gradient Propagation IG LRP DeepLIFT Grad-CAM and Attention
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Riemann sum approximation of Integrated Gradients (Sundararajan et al., 2017).
pub fn integrated_gradients(
    x: &Tensor,
    x_baseline: &Tensor,
    interpolated_grads: &Tensor,
) -> Result<Tensor> {
    let diff = (x - x_baseline)?;
    let avg_grad = interpolated_grads.mean(0)?;
    diff.mul(&avg_grad)
}

/// Layer-wise Relevance Propagation epsilon-rule for linear layer.
pub fn lrp_epsilon_rule(
    activations: &Tensor,
    weights: &Tensor,
    relevance_next: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    let d_in = activations.dim(0)?;
    let d_out = weights.dim(1)?;
    let a = activations.to_vec1::<f32>()?;
    let w = weights.to_vec2::<f32>()?;
    let r_next = relevance_next.to_vec1::<f32>()?;

    let mut z_j = vec![0.0f32; d_out];
    for j in 0..d_out {
        for i in 0..d_in {
            z_j[j] += a[i] * w[i][j];
        }
    }

    let mut s = vec![0.0f32; d_out];
    for j in 0..d_out {
        let sign = if z_j[j] >= 0.0 { 1.0f32 } else { -1.0f32 };
        s[j] = r_next[j] / (z_j[j] + epsilon * sign);
    }

    let mut r_i = vec![0.0f32; d_in];
    for i in 0..d_in {
        for j in 0..d_out {
            r_i[i] += a[i] * w[i][j] * s[j];
        }
    }

    Tensor::from_vec(r_i, (d_in,), activations.device())
}

/// Grad-CAM attribution for 2D spatial feature activations.
pub fn grad_cam_2d(activations: &Tensor, grads: &Tensor) -> Result<Tensor> {
    let c = activations.dim(0)?;
    let h = activations.dim(1)?;
    let w = activations.dim(2)?;
    let act_flat = activations.flatten_all()?.to_vec1::<f32>()?;
    let grad_flat = grads.flatten_all()?.to_vec1::<f32>()?;
    let hw = (h * w) as f32;

    let mut alpha = vec![0.0f32; c];
    for ch in 0..c {
        let mut sum = 0.0f32;
        for p in 0..(h * w) {
            sum += grad_flat[ch * (h * w) + p];
        }
        alpha[ch] = sum / hw;
    }

    let mut cam = vec![0.0f32; h * w];
    for p in 0..(h * w) {
        let mut sum = 0.0f32;
        for ch in 0..c {
            sum += alpha[ch] * act_flat[ch * (h * w) + p];
        }
        cam[p] = sum.max(0.0);
    }

    Tensor::from_vec(cam, (h, w), activations.device())
}

/// Adebayo parameter randomization test alignment metric.
pub fn cosine_similarity(s_orig: &Tensor, s_rand: &Tensor, eps: f32) -> Result<Tensor> {
    let orig = s_orig.flatten_all()?.to_vec1::<f32>()?;
    let rand = s_rand.flatten_all()?.to_vec1::<f32>()?;

    let mut dot = 0.0f32;
    let mut norm_o = 0.0f32;
    let mut norm_r = 0.0f32;
    for i in 0..orig.len() {
        dot += orig[i] * rand[i];
        norm_o += orig[i] * orig[i];
        norm_r += rand[i] * rand[i];
    }

    let sim = dot / (norm_o.sqrt() * norm_r.sqrt() + eps);
    Tensor::new(sim, s_orig.device())
}
