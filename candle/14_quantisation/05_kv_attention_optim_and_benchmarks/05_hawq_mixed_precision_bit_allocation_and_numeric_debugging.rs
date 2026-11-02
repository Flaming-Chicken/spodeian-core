//! HAWQ Mixed Precision Bit Allocation and Numeric Debugging
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Symmetric per-row uniform quantization to specified integer bitwidth.
pub fn quantize_weight_bits(w: &Tensor, bits: usize) -> Result<Tensor> {
    let levels = ((1usize << (bits - 1)) - 1) as f64;
    let max_abs = w.abs()?.max_keepdim(1)?;
    let scale = (max_abs / levels)?;
    let scale_eps = scale.affine(1.0, 1e-8)?;
    let normalized = w.broadcast_div(&scale_eps)?;
    let rounded = normalized.round()?;

    let min_bound = Tensor::new(-levels as f32, w.device())?.broadcast_as(w.shape())?;
    let max_bound = Tensor::new(levels as f32, w.device())?.broadcast_as(w.shape())?;
    let q = rounded.clamp(&min_bound, &max_bound)?;
    q.broadcast_mul(&scale)
}

/// Computes second-order perturbation sensitivity Omega(b) = 0.5 * Tr(H) * ||W - W_b||^2.
pub fn compute_layer_sensitivity(
    w: &Tensor,
    trace: f64,
    candidate_bits: &[usize],
) -> Result<HashMap<usize, f64>> {
    let mut sensitivities = HashMap::new();
    for &b in candidate_bits {
        let w_q = quantize_weight_bits(w, b)?;
        let diff = (w - &w_q)?;
        let err = diff.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        sensitivities.insert(b, 0.5 * trace * err);
    }
    Ok(sensitivities)
}

/// Greedy Pareto / ILP bit allocation under parameter bit budget.
pub fn greedy_hawq_bit_allocation(
    layer_weights: &[Tensor],
    layer_traces: &[f64],
    target_avg_bits: f64,
    candidate_bits: &[usize],
) -> Result<Vec<usize>> {
    let numels: Vec<usize> = layer_weights.iter().map(|w| w.elem_count()).collect();
    let total_params: usize = numels.iter().sum();
    let max_total_bits = target_avg_bits * (total_params as f64);

    let mut sensitivities = Vec::new();
    for (w, &tr) in layer_weights.iter().zip(layer_traces.iter()) {
        sensitivities.push(compute_layer_sensitivity(w, tr, candidate_bits)?);
    }

    let min_b = *candidate_bits.iter().min().unwrap();
    let mut alloc = vec![min_b; layer_weights.len()];

    let current_total_bits = |curr_alloc: &[usize]| -> f64 {
        curr_alloc
            .iter()
            .zip(numels.iter())
            .map(|(&b, &n)| (b * n) as f64)
            .sum()
    };

    loop {
        let mut best_layer = None;
        let mut best_gain_ratio = -1.0f64;
        let mut best_next_bit = None;

        for l in 0..alloc.len() {
            let cur_b = alloc[l];
            if let Some(idx) = candidate_bits.iter().position(|&x| x == cur_b) {
                if idx + 1 < candidate_bits.len() {
                    let next_b = candidate_bits[idx + 1];
                    let delta_cost = ((next_b - cur_b) * numels[l]) as f64;
                    if current_total_bits(&alloc) + delta_cost <= max_total_bits {
                        let gain = sensitivities[l][&cur_b] - sensitivities[l][&next_b];
                        let ratio = gain / delta_cost;
                        if ratio > best_gain_ratio {
                            best_gain_ratio = ratio;
                            best_layer = Some(l);
                            best_next_bit = Some(next_b);
                        }
                    }
                }
            }
        }

        match (best_layer, best_next_bit) {
            (Some(layer_idx), Some(next_bit)) => {
                alloc[layer_idx] = next_bit;
            }
            _ => break,
        }
    }

    Ok(alloc)
}

/// Runs dual FP32 and quantized forward passes and reports layerwise SQNR in dB.
pub fn layerwise_sqnr_diagnostic(
    x: &Tensor,
    weights: &[Tensor],
    bit_allocation: &[usize],
) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
    let mut a_fp = x.clone();
    let mut a_quant = x.clone();
    let mut fp_acts = Vec::new();
    let mut q_acts = Vec::new();
    let mut quant_weights = Vec::new();
    let mut sqnr_vals = Vec::new();

    for (w, &b) in weights.iter().zip(bit_allocation.iter()) {
        let w_q = quantize_weight_bits(w, b)?;
        quant_weights.push(w_q.clone());

        a_fp = a_fp.matmul(w)?;
        a_quant = a_quant.matmul(&w_q)?;

        fp_acts.push(a_fp.clone());
        q_acts.push(a_quant.clone());

        let sig = a_fp.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let noise = (&a_fp - &a_quant)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let sqnr = 10.0 * (sig / (noise + 1e-12)).log10();
        sqnr_vals.push(sqnr as f32);
    }

    let sqnr_tensor = Tensor::from_vec(sqnr_vals, (weights.len(),), x.device())?;
    Ok((sqnr_tensor, quant_weights, fp_acts, q_acts))
}
