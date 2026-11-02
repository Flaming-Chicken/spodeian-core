//! AOT Inductor and ExecuTorch Edge Concepts in candle/17_deployment_low_level_and_capstones.

use candle_core::{Device, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Edge operator composition: Linear -> LayerNorm -> SiLU.
pub fn aot_edge_forward(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let z = x.matmul(w)?.broadcast_add(b)?;
    let mu = z.mean_keepdim(1)?;
    let centered = z.broadcast_sub(&mu)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let denom = (var + 1e-5)?.sqrt()?;
    let norm = centered.broadcast_div(&denom)?;
    let hat_z = norm.broadcast_mul(gamma)?.broadcast_add(beta)?;
    let y = candle_nn::ops::silu(&hat_z)?;
    Ok((y, z, hat_z))
}

/// Simulates AOTInductor / ExecuTorch static memory arena planning with buffer reuse.
pub fn plan_static_memory_arena(
    tensor_bytes: &[usize],
    reuse_pairs: &[(usize, usize)],
    device: &Device,
) -> Result<Tensor> {
    let mut offsets = vec![0i64; tensor_bytes.len()];
    let mut current_offset = 0i64;
    let mut reused = std::collections::HashMap::new();

    for &(src, dst) in reuse_pairs {
        reused.insert(dst, src);
    }

    for (i, &size) in tensor_bytes.iter().enumerate() {
        if let Some(&src) = reused.get(&i) {
            offsets[i] = offsets[src];
        } else {
            offsets[i] = current_offset;
            current_offset += size as i64;
        }
    }

    Tensor::from_slice(&offsets, (tensor_bytes.len(),), device)
}
