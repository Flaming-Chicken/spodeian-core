//! Neural Audio Codecs Encodec RVQ and Quantizer Dropout
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Snake periodic activation function: Snake(x) = x + (1 / alpha) * sin^2(alpha * x).
pub fn snake_activation(x: &Tensor, alpha: f64) -> Result<Tensor> {
    let ax = (x * alpha)?;
    let sin_ax = ax.sin()?;
    let sin_sq = sin_ax.sqr()?;
    let term = (sin_sq * (1.0 / alpha))?;
    x.add(&term)
}

/// Multi-scale Residual Vector Quantization (RVQ) with quantizer dropout.
pub fn residual_vector_quantization(
    z: &Tensor,
    codebooks: &Tensor,
    n_active: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (n_q, _k_entries, d_dim) = codebooks.dims3()?;
    let b_dim = z.dim(0)?;

    let mut residual = z.clone();
    let mut z_q = Tensor::zeros((b_dim, d_dim), z.dtype(), z.device())?;
    let mut all_indices = Vec::with_capacity(n_q);

    for k in 0..n_q {
        let cb_k = codebooks.get(k)?; // [K, D]
        let r_sq = residual.sqr()?.sum_keepdim(1)?; // [B, 1]
        let dot = residual.matmul(&cb_k.t()?)?; // [B, K]
        let cb_sq = cb_k.sqr()?.sum_keepdim(1)?.t()?; // [1, K]

        let dists = r_sq.broadcast_sub(&(dot * 2.0)?)?.broadcast_add(&cb_sq)?; // [B, K]

        let idx = dists.argmin(1)?; // [B] of U32
        all_indices.push(idx.to_dtype(candle_core::DType::F32)?);

        let quantized_k = cb_k.index_select(&idx, 0)?; // [B, D]

        if k < n_active {
            z_q = z_q.add(&quantized_k)?;
        }
        residual = residual.sub(&quantized_k)?;
    }

    let indices_tensor = Tensor::stack(&all_indices, 0)?;
    Ok((z_q, indices_tensor, residual))
}
