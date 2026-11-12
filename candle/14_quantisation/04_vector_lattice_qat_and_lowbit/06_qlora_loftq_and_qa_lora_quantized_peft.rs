//! QLoRA, LoftQ, and QA-LoRA Quantized PEFT
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Canonical QLoRA NF4 16-level standard codebook (Dettmers et al. 2023).
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Information-theoretically optimal NormalFloat 4 (NF4) block quantizer.
pub fn nf4_quantize(x: &Tensor, block_size: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut deq_vec = Vec::with_capacity(vec.len());
    let mut idx_vec = Vec::with_capacity(vec.len());
    let mut scales_vec = Vec::new();

    for chunk in vec.chunks(block_size) {
        let abs_max = chunk
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()))
            .max(1e-7);
        scales_vec.push(abs_max);
        let inv_scale = 1.0f32 / abs_max;

        for &v in chunk {
            let norm_v = v * inv_scale;
            let mut best_idx = 0;
            let mut best_lvl = NF4_LEVELS[0];
            let mut min_diff = (norm_v - best_lvl).abs();

            for (i, &lvl) in NF4_LEVELS[iter_range()].iter().enumerate() {
                let diff = (norm_v - lvl).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_lvl = lvl;
                    best_idx = i;
                }
            }
            idx_vec.push(best_idx as f32);
            deq_vec.push(best_lvl * abs_max);
        }
    }

    let n_scales = scales_vec.len();
    let deq = Tensor::from_vec(deq_vec, x.shape(), x.device())?;
    let idx = Tensor::from_vec(idx_vec, x.shape(), x.device())?;
    let scales = Tensor::from_vec(scales_vec, &[n_scales], x.device())?;
    Ok((deq, idx, scales))
}

fn iter_range() -> std::ops::Range<usize> {
    0..16
}

/// QLoRA Double Quantization: quantize FP32 scales c1 to 8-bit integers.
pub fn double_quantize(c1: &Tensor, dq_block_size: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let vec = c1.flatten_all()?.to_vec1::<f32>()?;
    let mut c1_deq_vec = Vec::with_capacity(vec.len());
    let mut q2_vec = Vec::with_capacity(vec.len());
    let mut c2_vec = Vec::new();

    for chunk in vec.chunks(dq_block_size) {
        let max_val = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v));
        let c2 = (max_val / 127.0f32).max(1e-7);
        c2_vec.push(c2);
        let inv_c2 = 1.0f32 / c2;

        for &v in chunk {
            let q = (v * inv_c2).round().clamp(-127.0f32, 127.0f32);
            q2_vec.push(q);
            c1_deq_vec.push(q * c2);
        }
    }

    let n_c2 = c2_vec.len();
    let c1_deq = Tensor::from_vec(c1_deq_vec, c1.shape(), c1.device())?;
    let q2 = Tensor::from_vec(q2_vec, c1.shape(), c1.device())?;
    let c2 = Tensor::from_vec(c2_vec, &[n_c2], c1.device())?;
    Ok((c1_deq, q2, c2))
}

/// QLoRA forward pass: Base NF4 frozen GEMM + low-rank LoRA adapter.
pub fn qlora_forward(
    x: &Tensor,
    w_deq: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    scaling: f64,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let base_out = x.matmul(w_deq)?;
    let lora_out = (x.matmul(lora_a)?.matmul(lora_b)? * scaling)?;
    let mut y = (base_out + lora_out)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok(y)
}

/// QA-LoRA forward pass: group-wise quantization-aware low-rank adaptation.
pub fn qa_lora_forward(
    x: &Tensor,
    w_deq: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    group_scales: &Tensor,
    scaling: f64,
    group_size: usize,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let g_vec = group_scales.flatten_all()?.to_vec1::<f32>()?;
    let mut exp = Vec::with_capacity(g_vec.len() * group_size);
    for &s in &g_vec {
        for _ in 0..group_size {
            exp.push(s);
        }
    }
    let g_exp = Tensor::from_vec(exp, &[g_vec.len() * group_size, 1], x.device())?;
    let a_qa = lora_a.broadcast_mul(&g_exp)?;
    let base_out = x.matmul(w_deq)?;
    let lora_out = (x.matmul(&a_qa)?.matmul(lora_b)? * scaling)?;
    let mut y = (base_out + lora_out)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok((y, a_qa))
}
