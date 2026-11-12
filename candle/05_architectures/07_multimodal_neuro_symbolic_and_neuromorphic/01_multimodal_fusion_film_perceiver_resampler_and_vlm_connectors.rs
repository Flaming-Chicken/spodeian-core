//! Multimodal Fusion FiLM Perceiver Resampler and VLM Connectors
//! in candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Feature-wise Linear Modulation: gamma * x + beta
pub fn film_modulation(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    gamma.broadcast_mul(x)?.broadcast_add(beta)
}

fn linear_3d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let b = dims[0];
    let l = dims[1];
    let d = dims[2];
    let d_out = w.dims()[1];
    let x_flat = x.reshape((b * l, d))?;
    let y_flat = x_flat.matmul(w)?;
    y_flat.reshape((b, l, d_out))
}

/// Perceiver Resampler cross-attention: maps variable visual tokens to fixed K latents.
pub fn perceiver_resampler(
    latent_queries: &Tensor,
    visual_tokens: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
) -> Result<Tensor> {
    let dims = latent_queries.dims();
    let d = dims[dims.len() - 1];
    let q = linear_3d(latent_queries, w_q)?;
    let k = linear_3d(visual_tokens, w_k)?;
    let v = linear_3d(visual_tokens, w_v)?;

    let k_t = k.transpose(1, 2)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scores = (q.matmul(&k_t)? * scale)?;
    let attn = candle_nn::ops::softmax(&scores, 2)?;
    attn.matmul(&v)
}

/// VLM spatial pool connector: compresses adjacent patches into an LLM token via 2-layer MLP.
pub fn vlm_spatial_pool_connector(
    patches: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let dims = patches.dims();
    let b = dims[0];
    let num_patches = dims[1];
    let d = dims[2];

    let unshuffled = patches.reshape((b, num_patches * d))?;
    let h = unshuffled.matmul(w1)?.broadcast_add(b1)?.gelu_erf()?;
    let out = h.matmul(w2)?.broadcast_add(b2)?;
    let d_llm = out.dims()[1];
    out.reshape((b, 1, d_llm))
}
