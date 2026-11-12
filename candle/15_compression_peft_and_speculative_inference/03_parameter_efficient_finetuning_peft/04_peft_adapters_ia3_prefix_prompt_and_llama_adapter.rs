//! PEFT Adapters Ia3 Prefix Prompt and Llama Adapter
//! in candle/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// (IA)^3 multiplicative rescaling: activation * scale.
pub fn ia3_rescale(activation: &Tensor, scale: &Tensor) -> Result<Tensor> {
    activation.broadcast_mul(scale)
}

/// Prefix-Tuning attention with prepended key and value prefixes.
pub fn prefix_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
) -> Result<Tensor> {
    let k_cat = Tensor::cat(&[prefix_k, k], 1)?;
    let v_cat = Tensor::cat(&[prefix_v, v], 1)?;
    let d = q.dim(2)?;
    let scale = 1.0 / (d as f64).sqrt();

    let scores = (q.matmul(&k_cat.transpose(1, 2)?)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    attn.matmul(&v_cat)
}

/// LLaMA-Adapter zero-gated prefix attention.
pub fn llama_adapter_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    let k_cat = Tensor::cat(&[prefix_k, k], 1)?;
    let d = q.dim(2)?;
    let scale = 1.0 / (d as f64).sqrt();

    let scores = (q.matmul(&k_cat.transpose(1, 2)?)? * scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;

    let prefix_len = prefix_k.dim(1)?;
    let seq_len = k.dim(1)?;

    let attn_prefix = attn.narrow(2, 0, prefix_len)?;
    let attn_seq = attn.narrow(2, prefix_len, seq_len)?;

    let seq_out = attn_seq.matmul(v)?;
    let prefix_out = attn_prefix.matmul(prefix_v)?;
    let gated_prefix = prefix_out.broadcast_mul(gate)?;

    seq_out.add(&gated_prefix)
}
