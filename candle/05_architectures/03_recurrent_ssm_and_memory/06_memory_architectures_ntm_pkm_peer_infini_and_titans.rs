//! Memory Architectures: NTM, PKM, PEER, Infini-attention, and Titans
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// NTM content-based addressing and read head.
pub fn ntm_content_addressing_read(
    memory: &Tensor,
    key: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // key norm: [B, W] -> [B, 1, W]
    let key_norm = key
        .broadcast_div(&key.sqr()?.sum_keepdim(1)?.sqrt()?)?
        .unsqueeze(1)?;
    // mem norm: [B, N, W]
    let mem_norm = memory.broadcast_div(&memory.sqr()?.sum_keepdim(2)?.sqrt()?)?;

    let cos_sim = key_norm.broadcast_mul(&mem_norm)?.sum(2)?; // [B, N]
    let scaled = cos_sim.broadcast_mul(beta)?; // [B, N]

    let weights = candle_nn::ops::softmax(&scaled, 1)?; // [B, N]
    let read_vec = weights.unsqueeze(1)?.matmul(memory)?.squeeze(1)?; // [B, W]

    Ok((read_vec, weights))
}

/// Non-negative ELU kernel: phi(x) = elu(x) + 1.0 = if x > 0 { x + 1 } else { exp(x) }.
pub fn phi_elu(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape();
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(flat.len());
    for v in flat {
        if v > 0.0 {
            out.push(v + 1.0);
        } else {
            out.push(v.exp());
        }
    }
    Tensor::from_vec(out, dims, x.device())
}

/// Infini-attention compressive memory retrieval and update step.
pub fn infini_memory_step(
    m_prev: &Tensor,
    z_prev: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let phi_q = phi_elu(q)?; // [B, D]
    let phi_k = phi_elu(k)?; // [B, D]

    // Retrieval
    let num = phi_q.unsqueeze(1)?.matmul(m_prev)?.squeeze(1)?; // [B, D]
    let denom = (phi_q.mul(z_prev)?.sum_keepdim(1)? + 1e-4)?; // [B, 1]
    let a_mem = num.broadcast_div(&denom)?; // [B, D]

    // Update
    let m_delta = phi_k.unsqueeze(2)?.matmul(&v.unsqueeze(1)?)?; // [B, D, D]
    let m_next = (m_prev + m_delta)?;
    let z_next = (z_prev + phi_k)?;

    Ok((a_mem, m_next, z_next))
}
