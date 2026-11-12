//! Vector Product and Additive Quantization VQ PQ RVQ AQLM
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Single-codebook Vector Quantization (VQ): nearest Euclidean centroid.
///
/// vectors: [N, D]
/// codebook: [K, D]
/// returns: (recon [N, D], indices [N] as F32)
pub fn vector_quantize(vectors: &Tensor, codebook: &Tensor) -> Result<(Tensor, Tensor)> {
    let vec_sq = vectors.sqr()?.sum_keepdim(1)?;
    let cb_sq = codebook.sqr()?.sum_keepdim(1)?.t()?;
    let dot = vectors.matmul(&codebook.t()?)?;
    let two_dot = (dot * 2.0)?;
    let dist = vec_sq.broadcast_sub(&two_dot)?.broadcast_add(&cb_sq)?;

    let indices = dist.argmin(1)?;
    let recon = codebook.index_select(&indices, 0)?;
    let indices_f32 = indices.to_dtype(DType::F32)?;
    Ok((recon, indices_f32))
}

/// Product Quantization (PQ): splits D-dim vector into M subvectors.
///
/// vectors: [N, D]
/// codebooks: [M, K, D_sub] where D = M * D_sub
/// returns: (recon [N, D], indices [N, M] as F32)
pub fn product_quantize(vectors: &Tensor, codebooks: &Tensor) -> Result<(Tensor, Tensor)> {
    let (_n, d) = vectors.dims2()?;
    let (m, _k, d_sub) = codebooks.dims3()?;
    if d != m * d_sub {
        candle_core::bail!("Dimension mismatch in PQ: {} != {} * {}", d, m, d_sub);
    }

    let mut recon_subs = Vec::with_capacity(m);
    let mut indices_subs = Vec::with_capacity(m);

    for i in 0..m {
        let sub_vec = vectors.narrow(1, i * d_sub, d_sub)?;
        let sub_cb = codebooks.get(i)?;
        let (sub_recon, sub_idx) = vector_quantize(&sub_vec, &sub_cb)?;
        recon_subs.push(sub_recon);
        indices_subs.push(sub_idx.unsqueeze(1)?);
    }

    let recon = Tensor::cat(&recon_subs, 1)?;
    let indices = Tensor::cat(&indices_subs, 1)?;
    Ok((recon, indices))
}

/// Residual Vector Quantization (RVQ): sequential multi-stage codebook refinement.
///
/// vectors: [N, D]
/// codebooks: [M, K, D]
/// returns: (recon [N, D], indices [N, M] as F32)
pub fn residual_vector_quantize(vectors: &Tensor, codebooks: &Tensor) -> Result<(Tensor, Tensor)> {
    let mut residual = vectors.clone();
    let mut recon = Tensor::zeros_like(vectors)?;
    let m = codebooks.dim(0)?;
    let mut indices_subs = Vec::with_capacity(m);

    for i in 0..m {
        let cb = codebooks.get(i)?;
        let (stage_recon, stage_idx) = vector_quantize(&residual, &cb)?;
        recon = (&recon + &stage_recon)?;
        residual = (&residual - &stage_recon)?;
        indices_subs.push(stage_idx.unsqueeze(1)?);
    }

    let indices = Tensor::cat(&indices_subs, 1)?;
    Ok((recon, indices))
}

/// Additive Multi-Codebook Quantization (AQLM) dequantization:
/// recon = sum_{m=0}^{M-1} codebooks[m][indices[:, m]]
pub fn aqlm_dequantize(indices: &Tensor, codebooks: &Tensor) -> Result<Tensor> {
    let (n, m) = indices.dims2()?;
    let (_, _, d) = codebooks.dims3()?;
    let mut recon = Tensor::zeros((n, d), DType::F32, indices.device())?;

    for i in 0..m {
        let stage_idx = indices.narrow(1, i, 1)?.squeeze(1)?.to_dtype(DType::U32)?;
        let cb = codebooks.get(i)?;
        let stage_recon = cb.index_select(&stage_idx, 0)?;
        recon = (&recon + &stage_recon)?;
    }
    Ok(recon)
}

/// AQLM forward pass: reconstructs [d_in, d_out] weight matrix and computes Y = X @ W + bias.
pub fn aqlm_forward(
    x: &Tensor,
    indices: &Tensor,
    codebooks: &Tensor,
    d_in: usize,
    d_out: usize,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let w_recon_vecs = aqlm_dequantize(indices, codebooks)?;
    let w_deq = w_recon_vecs.reshape((d_in, d_out))?;
    let mut y = x.matmul(&w_deq)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok((y, w_deq))
}
