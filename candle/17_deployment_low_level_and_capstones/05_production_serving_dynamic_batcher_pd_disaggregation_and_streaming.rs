//! Production Serving Dynamic Batcher Pd Disaggregation and Streaming
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for batched 3D tensor multiplied by 2D weight matrix: [B, S, D] x [D, H] -> [B, S, H].
pub fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    let h = w.dim(1)?;
    let flat_x = x.reshape((b * s, d))?;
    let flat_y = flat_x.matmul(w)?;
    flat_y.reshape((b, s, h))
}

/// Prefill worker: compute-bound full causal self-attention over prompt sequence.
pub fn prefill_step(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    bo: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (_b, s, d) = x.dims3()?;
    let q = matmul_3d_2d(x, wq)?;
    let k = matmul_3d_2d(x, wk)?;
    let v = matmul_3d_2d(x, wv)?;

    let k_t = k.transpose(1, 2)?;
    let scores = q.matmul(&k_t)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scaled_scores = scores.affine(scale, 0.0)?;

    // Upper triangular causal mask
    let mut mask_vec = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i {
                mask_vec[i * s + j] = -10000.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask_vec, (s, s), x.device())?;
    let masked_scores = scaled_scores.broadcast_add(&mask)?;

    let attn = candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;
    let context = attn.matmul(&v)?;
    let y = matmul_3d_2d(&context, wo)?.broadcast_add(bo)?;

    Ok((y, k, v))
}

/// Decode worker: memory-bound incremental step consuming KV-cache from prefill worker.
pub fn decode_step(
    x_new: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    bo: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (_b, _s, d) = x_new.dims3()?;
    let q_new = matmul_3d_2d(x_new, wq)?;
    let k_new = matmul_3d_2d(x_new, wk)?;
    let v_new = matmul_3d_2d(x_new, wv)?;

    let new_k = Tensor::cat(&[k_cache, &k_new], 1)?;
    let new_v = Tensor::cat(&[v_cache, &v_new], 1)?;

    let new_k_t = new_k.transpose(1, 2)?;
    let scores = q_new.matmul(&new_k_t)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scaled_scores = scores.affine(scale, 0.0)?;

    let attn = candle_nn::ops::softmax(&scaled_scores, candle_core::D::Minus1)?;
    let context = attn.matmul(&new_v)?;
    let y_token = matmul_3d_2d(&context, wo)?.broadcast_add(bo)?;

    Ok((y_token, new_k, new_v))
}
