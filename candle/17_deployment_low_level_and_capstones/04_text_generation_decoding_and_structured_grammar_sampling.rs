//! Text Generation Decoding and Structured Grammar Sampling
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

pub const MASK_VALUE: f32 = -10000.0;

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Applies multiplicative repetition penalty based on past token history.
pub fn apply_repetition_penalty(logits: &Tensor, history: &Tensor, penalty: f64) -> Result<Tensor> {
    let (b_size, v_size) = logits.dims2()?;
    let s_len = history.dim(1)?;
    let penalty = penalty as f32;

    let logits_vec = logits.to_vec2::<f32>()?;
    let history_vec = history.to_vec2::<f32>()?;

    let mut out_vec = logits_vec;
    for b in 0..b_size {
        for s in 0..s_len {
            let tok_idx = history_vec[b][s] as usize;
            if tok_idx < v_size {
                let val = out_vec[b][tok_idx];
                out_vec[b][tok_idx] = if val > 0.0 {
                    val / penalty
                } else {
                    val * penalty
                };
            }
        }
    }
    Tensor::new(out_vec, logits.device())
}

/// Pre-softmax grammar/FSM bitmask: invalid transition tokens are set to mask_value.
pub fn apply_grammar_mask(
    logits: &Tensor,
    grammar_mask: &Tensor,
    mask_value: f32,
) -> Result<Tensor> {
    let (b_size, v_size) = logits.dims2()?;
    let logits_vec = logits.to_vec2::<f32>()?;
    let mask_vec = grammar_mask.to_vec2::<f32>()?;

    let mut out_vec = vec![vec![0.0f32; v_size]; b_size];
    for b in 0..b_size {
        for v in 0..v_size {
            out_vec[b][v] = if mask_vec[b][v] > 0.5 {
                logits_vec[b][v]
            } else {
                mask_value
            };
        }
    }
    Tensor::new(out_vec, logits.device())
}

/// Scales logits by temperature T before softmax.
pub fn apply_temperature(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    let temp = (temperature as f64).max(1e-5);
    logits.affine(1.0 / temp, 0.0)
}

/// Keeps top-k logits per row, masking out lower logits to mask_value.
pub fn apply_top_k(logits: &Tensor, k: usize, mask_value: f32) -> Result<Tensor> {
    let (b_size, v_size) = logits.dims2()?;
    let logits_vec = logits.to_vec2::<f32>()?;

    let mut out_vec = vec![vec![0.0f32; v_size]; b_size];
    for b in 0..b_size {
        let mut sorted = logits_vec[b].clone();
        sorted.sort_by(|a, b_val| b_val.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let k_idx = k.min(v_size).saturating_sub(1);
        let k_th = sorted[k_idx];

        for v in 0..v_size {
            out_vec[b][v] = if logits_vec[b][v] >= k_th {
                logits_vec[b][v]
            } else {
                mask_value
            };
        }
    }
    Tensor::new(out_vec, logits.device())
}

/// Min-P sampling filter: retains tokens with p_i >= p_max * min_p, then renormalizes.
pub fn apply_min_p(probs: &Tensor, min_p: f32) -> Result<Tensor> {
    let (b_size, v_size) = probs.dims2()?;
    let probs_vec = probs.to_vec2::<f32>()?;

    let mut out_vec = vec![vec![0.0f32; v_size]; b_size];
    for b in 0..b_size {
        let mut p_max = 0.0f32;
        for v in 0..v_size {
            if probs_vec[b][v] > p_max {
                p_max = probs_vec[b][v];
            }
        }
        let threshold = p_max * min_p;

        let mut sum = 0.0f32;
        for v in 0..v_size {
            if probs_vec[b][v] >= threshold {
                out_vec[b][v] = probs_vec[b][v];
                sum += out_vec[b][v];
            } else {
                out_vec[b][v] = 0.0;
            }
        }
        let sum = sum.max(1e-12);
        for v in 0..v_size {
            out_vec[b][v] /= sum;
        }
    }
    Tensor::new(out_vec, probs.device())
}

/// Complete text generation decoding pipeline.
pub fn decoding_pipeline(
    logits: &Tensor,
    history: &Tensor,
    grammar_mask: &Tensor,
    temperature: f64,
    repetition_penalty: f64,
    top_k: usize,
    min_p: f32,
) -> Result<(Tensor, Tensor)> {
    let rep = apply_repetition_penalty(logits, history, repetition_penalty)?;
    let masked = apply_grammar_mask(&rep, grammar_mask, MASK_VALUE)?;
    let scaled = apply_temperature(&masked, temperature)?;
    let topk = apply_top_k(&scaled, top_k, MASK_VALUE)?;
    let probs = candle_nn::ops::softmax(&topk, candle_core::D::Minus1)?;
    let filtered_probs = apply_min_p(&probs, min_p)?;

    // Greedy token selection (argmax along dim -1)
    let (b_size, v_size) = filtered_probs.dims2()?;
    let p_vec = filtered_probs.to_vec2::<f32>()?;
    let mut greedy = vec![0.0f32; b_size];
    for b in 0..b_size {
        let mut best_idx = 0;
        let mut best_val = -1.0f32;
        for v in 0..v_size {
            if p_vec[b][v] > best_val {
                best_val = p_vec[b][v];
                best_idx = v;
            }
        }
        greedy[b] = best_idx as f32;
    }
    let greedy_tokens = Tensor::new(greedy, logits.device())?;

    Ok((filtered_probs, greedy_tokens))
}
