//! 2 4 Semi Structured Sparsity and Rigl Dynamic Sparse
//! in candle/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Applies NVIDIA Ampere 2:4 semi-structured sparsity pattern.
/// In each contiguous block of 4 elements, keeps the 2 with largest magnitude.
pub fn prune_2_to_4(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    if cols % 4 != 0 {
        candle_core::bail!("Columns ({cols}) must be divisible by 4");
    }
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let mut mask_vec = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..(cols / 4) {
            let offset = r * cols + b * 4;
            let block = [
                w_vec[offset].abs(),
                w_vec[offset + 1].abs(),
                w_vec[offset + 2].abs(),
                w_vec[offset + 3].abs(),
            ];
            let mut indexed: Vec<(usize, f32)> = block.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            mask_vec[offset + indexed[0].0] = 1.0;
            mask_vec[offset + indexed[1].0] = 1.0;
        }
    }

    let mask = Tensor::from_vec(mask_vec, (rows, cols), w.device())?;
    let w_sparse = w.broadcast_mul(&mask)?;
    Ok((w_sparse, mask))
}

/// Compresses 2:4 sparse weights into packed values and 2-bit index metadata.
pub fn compress_2_to_4(w: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let num_blocks = cols / 4;
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let m_vec = mask.flatten_all()?.to_vec1::<f32>()?;

    let mut vals = Vec::with_capacity(rows * num_blocks * 2);
    let mut idxs = Vec::with_capacity(rows * num_blocks * 2);

    for r in 0..rows {
        for b in 0..num_blocks {
            let offset = r * cols + b * 4;
            for i in 0..4 {
                if m_vec[offset + i] > 0.5 {
                    vals.push(w_vec[offset + i]);
                    idxs.push(i as f32);
                }
            }
        }
    }

    let values = Tensor::from_vec(vals, (rows, cols / 2), w.device())?;
    let indices = Tensor::from_vec(idxs, (rows, num_blocks * 2), w.device())?;
    Ok((values, indices))
}

/// RigL (Rigging the Lottery) dynamic sparse topology exploration step.
pub fn rigl_update_mask(
    w: &Tensor,
    grad: &Tensor,
    current_mask: &Tensor,
    drop_fraction: f64,
) -> Result<Tensor> {
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let g_vec = grad.flatten_all()?.to_vec1::<f32>()?;
    let m_vec = current_mask.flatten_all()?.to_vec1::<f32>()?;
    let total = w_vec.len();

    let mut active = Vec::new();
    let mut inactive = Vec::new();

    for i in 0..total {
        if m_vec[i] > 0.5 {
            active.push(i);
        } else {
            inactive.push(i);
        }
    }

    let k_drop = ((drop_fraction * active.len() as f64) as usize).max(1);

    // 1. Drop active with smallest |w|
    let mut active_pairs: Vec<(usize, f32)> = active.iter().map(|&i| (i, w_vec[i].abs())).collect();
    active_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let to_drop: Vec<usize> = active_pairs.iter().take(k_drop).map(|&(i, _)| i).collect();

    // 2. Regrow inactive with largest |grad|
    let mut inactive_pairs: Vec<(usize, f32)> =
        inactive.iter().map(|&i| (i, g_vec[i].abs())).collect();
    inactive_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let to_regrow: Vec<usize> = inactive_pairs
        .iter()
        .take(k_drop)
        .map(|&(i, _)| i)
        .collect();

    let mut new_m_vec = m_vec;
    for idx in to_drop {
        new_m_vec[idx] = 0.0;
    }
    for idx in to_regrow {
        new_m_vec[idx] = 1.0;
    }

    Tensor::from_vec(new_m_vec, current_mask.shape(), current_mask.device())
}
