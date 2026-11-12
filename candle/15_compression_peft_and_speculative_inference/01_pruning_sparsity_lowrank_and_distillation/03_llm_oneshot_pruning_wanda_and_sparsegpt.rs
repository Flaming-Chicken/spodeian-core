//! LLM Oneshot Pruning Wanda and Sparsegpt
//! in candle/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes column-wise L2 norm of calibration activations X.
pub fn compute_activation_norms(x: &Tensor) -> Result<Tensor> {
    x.sqr()?.sum(0)?.sqrt()
}

/// Wanda (Pruning by Weights and Activations) one-shot row-wise pruning.
pub fn wanda_prune(w: &Tensor, act_norms: &Tensor, sparsity: f64) -> Result<(Tensor, Tensor)> {
    let (d_out, d_in) = w.dims2()?;
    let k = ((sparsity * d_in as f64) as usize).max(1);
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let a_vec = act_norms.flatten_all()?.to_vec1::<f32>()?;

    let mut mask_vec = vec![0.0f32; d_out * d_in];
    for r in 0..d_out {
        let mut row_scores: Vec<(usize, f32)> = (0..d_in)
            .map(|c| (c, (w_vec[r * d_in + c].abs() * a_vec[c])))
            .collect();
        row_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let tau = row_scores[k - 1].1;
        for c in 0..d_in {
            let s = w_vec[r * d_in + c].abs() * a_vec[c];
            if s >= tau {
                mask_vec[r * d_in + c] = 1.0;
            }
        }
    }

    let mask = Tensor::from_vec(mask_vec, (d_out, d_in), w.device())?;
    let w_pruned = w.broadcast_mul(&mask)?;
    Ok((w_pruned, mask))
}

/// SparseGPT Optimal Brain Surgeon (OBS) one-shot pruning with column compensation.
pub fn sparsegpt_prune(w: &Tensor, h_inv: &Tensor, sparsity: f64) -> Result<(Tensor, Tensor)> {
    let (d_out, d_in) = w.dims2()?;
    let mut w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let h_vec = h_inv.flatten_all()?.to_vec1::<f32>()?;
    let mut mask_vec = vec![1.0f32; d_out * d_in];

    // Compute column norms and OBS scores
    let mut col_scores: Vec<(usize, f32)> = (0..d_in)
        .map(|c| {
            let mut sum_sq = 0.0f32;
            for r in 0..d_out {
                sum_sq += w_vec[r * d_in + c] * w_vec[r * d_in + c];
            }
            let norm = sum_sq.sqrt();
            let diag_val = h_vec[c * d_in + c].max(1e-8);
            (c, norm / diag_val.sqrt())
        })
        .collect();

    col_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let k = ((sparsity * d_in as f64) as usize).max(1);
    let mut prune_cols: Vec<usize> = col_scores.iter().take(k).map(|&(c, _)| c).collect();
    prune_cols.sort();

    for &j in &prune_cols {
        let denom = h_vec[j * d_in + j].max(1e-8);
        let mut col_w = vec![0.0f32; d_out];
        for r in 0..d_out {
            col_w[r] = w_vec[r * d_in + j];
            w_vec[r * d_in + j] = 0.0;
            mask_vec[r * d_in + j] = 0.0;
        }

        // Error compensation to columns j' > j
        for jp in (j + 1)..d_in {
            let h_val = h_vec[j * d_in + jp];
            for r in 0..d_out {
                w_vec[r * d_in + jp] -= (col_w[r] / denom) * h_val;
            }
        }
    }

    let w_pruned = Tensor::from_vec(w_vec, (d_out, d_in), w.device())?;
    let mask = Tensor::from_vec(mask_vec, (d_out, d_in), w.device())?;
    Ok((w_pruned, mask))
}

/// Relative Frobenius reconstruction error.
pub fn compute_reconstruction_error(x: &Tensor, w_orig: &Tensor, w_pruned: &Tensor) -> Result<f32> {
    let y_orig = x.matmul(&w_orig.t()?)?;
    let y_pruned = x.matmul(&w_pruned.t()?)?;
    let diff = (y_orig.broadcast_sub(&y_pruned))?
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?;
    let den = y_orig.sqr()?.sum_all()?.to_scalar::<f32>()? + 1e-8;
    Ok(diff / den)
}
