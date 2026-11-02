//! AI Watermarking AI Detection and Model Obfuscation
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Kirchenbauer et al. (2023) green-list logit biasing: l' = l + delta * m_green.
pub fn watermark_logit_bias(logits: &Tensor, green_mask: &Tensor, delta: f32) -> Result<Tensor> {
    let l_vec = logits.to_vec2::<f32>()?;
    let gm_vec = green_mask.to_vec1::<f32>()?;
    let b = l_vec.len();
    let v = l_vec[0].len();

    let mut out = vec![vec![0.0f32; v]; b];
    for i in 0..b {
        for j in 0..v {
            out[i][j] = l_vec[i][j] + delta * gm_vec[j];
        }
    }
    Tensor::new(out, logits.device())
}

/// One-tailed z-score hypothesis test for green token frequency (Kirchenbauer et al. 2023).
///
/// z = (|S_G| - \gamma T) / \sqrt{T \gamma (1 - \gamma) + \epsilon}
pub fn watermark_z_score(green_counts: &Tensor, seq_lens: &Tensor, gamma: f32) -> Result<Tensor> {
    let gc = green_counts.to_vec1::<f32>()?;
    let sl = seq_lens.to_vec1::<f32>()?;
    let b = gc.len();

    let mut z = vec![0.0f32; b];
    for i in 0..b {
        let exp = gamma * sl[i];
        let std = (sl[i] * gamma * (1.0 - gamma) + 1e-8f32).sqrt();
        z[i] = (gc[i] - exp) / std;
    }
    Tensor::new(z, green_counts.device())
}

/// Fast-DetectGPT (Bao et al. 2023) 1-pass conditional likelihood curvature score.
///
/// d = \frac{1}{T} \sum_{t=1}^T (\log p(x_t) - \log p_{\text{ref}}(x_t))
pub fn fast_detect_gpt_score(log_probs: &Tensor, ref_log_probs: &Tensor) -> Result<Tensor> {
    let lp = log_probs.to_vec2::<f32>()?;
    let rlp = ref_log_probs.to_vec2::<f32>()?;
    let b = lp.len();
    let t = lp[0].len();

    let mut scores = vec![0.0f32; b];
    for i in 0..b {
        let mut sum = 0.0f32;
        for j in 0..t {
            sum += lp[i][j] - rlp[i][j];
        }
        scores[i] = sum / (t as f32);
    }
    Tensor::new(scores, log_probs.device())
}

/// Passport Layer (Fan et al. 2019) secret key gating:
///
/// s = \sigma(K \cdot W_{\text{pass}}), \quad y = (x W + b) \odot s
pub fn passport_layer_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    passport_key: &Tensor,
    passport_weight: &Tensor,
) -> Result<Tensor> {
    let pk = passport_key.to_vec1::<f32>()?;
    let pw = passport_weight.to_vec2::<f32>()?;
    let d_k = pk.len();
    let d = pw[0].len();

    let mut scale = vec![0.0f32; d];
    for j in 0..d {
        let mut dot = 0.0f32;
        for k in 0..d_k {
            dot += pk[k] * pw[k][j];
        }
        scale[j] = 1.0f32 / (1.0f32 + (-dot).exp());
    }

    let base = x.matmul(weight)?.broadcast_add(bias)?;
    let base_vec = base.to_vec2::<f32>()?;
    let b = base_vec.len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            out[i][j] = base_vec[i][j] * scale[j];
        }
    }
    Tensor::new(out, x.device())
}
