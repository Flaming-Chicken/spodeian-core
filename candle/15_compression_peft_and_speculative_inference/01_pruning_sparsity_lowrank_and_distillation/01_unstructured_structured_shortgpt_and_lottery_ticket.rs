//! Unstructured Structured Shortgpt and Lottery Ticket
//! in candle/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Unstructured magnitude pruning.
pub fn unstructured_magnitude_prune(w: &Tensor, sparsity: f64) -> Result<(Tensor, Tensor)> {
    let k = ((sparsity * (w.elem_count() as f64)) as usize).max(1);
    let mut abs_vals = w.abs()?.flatten_all()?.to_vec1::<f32>()?;
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = abs_vals[k - 1];
    let mask = w.abs()?.ge(tau)?.to_dtype(candle_core::DType::F32)?;
    let w_pruned = w.broadcast_mul(&mask)?;
    Ok((w_pruned, mask))
}

/// Structured row/channel pruning based on L2 norm.
pub fn structured_row_prune(w: &Tensor, keep_ratio: f64) -> Result<(Tensor, Tensor)> {
    let (rows, _cols) = w.dims2()?;
    let k_rows = ((keep_ratio * (rows as f64)) as usize).max(1);
    let row_norms = w
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut indexed_norms: Vec<(usize, f32)> = row_norms.into_iter().enumerate().collect();
    indexed_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut top_indices: Vec<u32> = indexed_norms
        .iter()
        .take(k_rows)
        .map(|&(i, _)| i as u32)
        .collect();
    top_indices.sort();
    let idx_tensor = Tensor::from_vec(top_indices.clone(), (k_rows,), w.device())?;
    let w_sliced = w.index_select(&idx_tensor, 0)?;
    let indices_f32 = Tensor::from_vec(
        top_indices.into_iter().map(|x| x as f32).collect(),
        (k_rows,),
        w.device(),
    )?;
    Ok((w_sliced, indices_f32))
}

/// ShortGPT Block Influence (BI): 1 - mean(cos(x_in, x_out)).
pub fn compute_block_influence(x_in: &Tensor, x_out: &Tensor) -> Result<f32> {
    let dot = (x_in * x_out)?.sum_keepdim(1)?;
    let norm_in = x_in.sqr()?.sum_keepdim(1)?.sqrt()?;
    let norm_out = x_out.sqr()?.sum_keepdim(1)?.sqrt()?;
    let denom = (norm_in * norm_out)?.affine(1.0, 1e-8)?;
    let cos = (dot / denom)?;
    let mean_cos = cos.mean_all()?.to_scalar::<f32>()?;
    Ok(1.0 - mean_cos)
}

/// Lottery ticket subnetwork extraction from trained mask applied to initial weights.
pub fn lottery_ticket_rewind(
    w_initial: &Tensor,
    w_trained: &Tensor,
    sparsity: f64,
) -> Result<(Tensor, Tensor)> {
    let k = ((sparsity * (w_trained.elem_count() as f64)) as usize).max(1);
    let mut abs_vals = w_trained.abs()?.flatten_all()?.to_vec1::<f32>()?;
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = abs_vals[k - 1];
    let mask = w_trained
        .abs()?
        .ge(tau)?
        .to_dtype(candle_core::DType::F32)?;
    let w_subnetwork = w_initial.broadcast_mul(&mask)?;
    Ok((w_subnetwork, mask))
}
