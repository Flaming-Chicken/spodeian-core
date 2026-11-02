//! Audio Dsp Spectrograms Conformer Ctc and Vocoder
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Projects linear frequency magnitude spectrum to Mel scale and computes log power.
/// magnitude_spec: [B, F, T]
/// mel_basis: [M, F]
/// Returns: [B, M, T]
pub fn log_mel_spectrogram(
    magnitude_spec: &Tensor,
    mel_basis: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let mel_basis_expanded = mel_basis.unsqueeze(0)?;
    let mel_spec = mel_basis_expanded.broadcast_matmul(magnitude_spec)?;
    (mel_spec + eps)?.log()
}

/// Conformer Convolution Module
pub fn conformer_conv_module(
    x: &Tensor,
    w_p1: &Tensor,
    b_p1: &Tensor,
    w_dw: &Tensor,
    b_dw: &Tensor,
    w_p2: &Tensor,
    b_p2: &Tensor,
) -> Result<Tensor> {
    let (b_sz, t_sz, d_sz) = x.dims3()?;

    // 1. LayerNorm
    let mean = x.mean_keepdim(2)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(2)?;
    let std = (var + 1e-5)?.sqrt()?;
    let x_norm = diff.broadcast_div(&std)?;

    // 2. Pointwise conv 1: [B, T, D] @ [D, 2*D] + [2*D]
    let x_flat = x_norm.reshape((b_sz * t_sz, d_sz))?;
    let proj1 = x_flat
        .matmul(w_p1)?
        .reshape((b_sz, t_sz, 2 * d_sz))?
        .broadcast_add(b_p1)?;
    let a = proj1.narrow(2, 0, d_sz)?;
    let b = proj1.narrow(2, d_sz, d_sz)?;
    let glu = (a * candle_nn::ops::sigmoid(&b)?)?;

    // 3. Depthwise 1D conv: kernel 3, padding 1
    let pad = Tensor::zeros((b_sz, 1, d_sz), x.dtype(), x.device())?;
    let padded = Tensor::cat(&[&pad, &glu, &pad], 1)?;
    let t0 = padded.narrow(1, 0, t_sz)?;
    let t1 = padded.narrow(1, 1, t_sz)?;
    let t2 = padded.narrow(1, 2, t_sz)?;

    let k0 = w_dw
        .narrow(2, 0, 1)?
        .squeeze(2)?
        .squeeze(1)?
        .reshape((1, 1, d_sz))?;
    let k1 = w_dw
        .narrow(2, 1, 1)?
        .squeeze(2)?
        .squeeze(1)?
        .reshape((1, 1, d_sz))?;
    let k2 = w_dw
        .narrow(2, 2, 1)?
        .squeeze(2)?
        .squeeze(1)?
        .reshape((1, 1, d_sz))?;
    let b_conv = b_dw.reshape((1, 1, d_sz))?;

    let dw_out = ((t0.broadcast_mul(&k0)? + t1.broadcast_mul(&k1)?)? + t2.broadcast_mul(&k2)?)?
        .broadcast_add(&b_conv)?;

    // 4. SiLU activation
    let act = candle_nn::ops::silu(&dw_out)?;

    // 5. Pointwise conv 2: [B, T, D] @ [D, D] + [D]
    let act_flat = act.reshape((b_sz * t_sz, d_sz))?;
    let out = act_flat
        .matmul(w_p2)?
        .reshape((b_sz, t_sz, d_sz))?
        .broadcast_add(b_p2)?;

    // 6. Residual
    x + out
}

/// Helper for LogSumExp of a slice of floats
fn logsumexp(vals: &[f32]) -> f32 {
    let max_v = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let sum: f32 = vals.iter().map(|&v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Computes exact CTC loss for a single sequence using dynamic programming forward variables.
pub fn ctc_forward_loss(log_probs: &Tensor, target: &[u32], blank: u32) -> Result<Tensor> {
    let (t_len, _v_len) = log_probs.dims2()?;
    let lp = log_probs.to_vec2::<f32>()?;

    let u_len = target.len();
    let s_len = 2 * u_len + 1;
    let mut target_ext = vec![blank; s_len];
    for i in 0..u_len {
        target_ext[2 * i + 1] = target[i];
    }

    let neg_inf = -1e9f32;
    let mut alpha = vec![neg_inf; s_len];
    alpha[0] = lp[0][target_ext[0] as usize];
    if s_len > 1 {
        alpha[1] = lp[0][target_ext[1] as usize];
    }

    for t in 1..t_len {
        let prev_alpha = alpha.clone();
        for s in 0..s_len {
            let sym = target_ext[s];
            let mut terms = vec![prev_alpha[s]];
            if s > 0 {
                terms.push(prev_alpha[s - 1]);
            }
            if s >= 2 && sym != blank && sym != target_ext[s - 2] {
                terms.push(prev_alpha[s - 2]);
            }
            let lse = logsumexp(&terms);
            alpha[s] = lse + lp[t][sym as usize];
        }
    }

    let final_loss = -logsumexp(&[alpha[s_len - 1], alpha[s_len - 2]]);
    Tensor::new(final_loss, log_probs.device())
}

/// Multi-Resolution STFT spectral convergence and log-magnitude loss.
pub fn mrstft_loss(y_pred: &Tensor, y_true: &Tensor, eps: f64) -> Result<Tensor> {
    let diff = (y_true - y_pred)?;
    let fro_diff = (diff.sqr()?.sum_keepdim(1)?.sum_keepdim(2)? + eps)?.sqrt()?;
    let fro_true = (y_true.sqr()?.sum_keepdim(1)?.sum_keepdim(2)? + eps)?.sqrt()?;
    let sc_loss = fro_diff.div(&fro_true)?.mean_all()?;

    let log_true = (y_true + eps)?.log()?;
    let log_pred = (y_pred + eps)?.log()?;
    let mag_loss = (log_true - log_pred)?.abs()?.mean_all()?;

    sc_loss + mag_loss
}
