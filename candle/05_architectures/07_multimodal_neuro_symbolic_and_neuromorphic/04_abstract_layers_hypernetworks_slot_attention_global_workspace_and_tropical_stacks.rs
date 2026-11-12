//! Abstract Layers: Hypernetworks, Slot Attention, Global Workspace, and Tropical Stacks
//! in candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Hypernetwork linear layer: generates dynamic weights from conditioning code z.
pub fn hypernetwork_linear(
    x: &Tensor,
    z: &Tensor,
    w_h: &Tensor,
    b_h: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let x_dims = x.dims();
    let b = x_dims[0];
    let d_in = x_dims[1];
    let d_out = bias.dims()[0];

    let w_flat = z.matmul(w_h)?.broadcast_add(b_h)?;
    let w_dyn = w_flat.reshape((b, d_in, d_out))?;
    let x_exp = x.unsqueeze(1)?;
    let y = x_exp.matmul(&w_dyn)?.squeeze(1)?.broadcast_add(bias)?;
    Ok(y)
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

/// Slot Attention: competitive slot attention over input tokens.
pub fn slot_attention(
    slots: &Tensor,
    inputs: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
) -> Result<Tensor> {
    let d = slots.dims()[2];
    let q = linear_3d(slots, w_q)?;
    let k = linear_3d(inputs, w_k)?;
    let v = linear_3d(inputs, w_v)?;

    let k_t = k.transpose(1, 2)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scores = (q.matmul(&k_t)? * scale)?;
    // Softmax over slots (dim 1)
    let attn = candle_nn::ops::softmax(&scores, 1)?;
    let attn_sum = (attn.sum_keepdim(2)? + 1e-8f64)?;
    let attn_norm = attn.broadcast_div(&attn_sum)?;
    let updates = attn_norm.matmul(&v)?;
    slots.add(&updates)
}

/// Tropical (max-plus) semiring matrix multiplication: Y_ij = max_k (X_ik + W_kj).
pub fn tropical_max_plus_matmul(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let x_exp = x.unsqueeze(2)?;
    let w_exp = w.unsqueeze(0)?;
    let summed = x_exp.broadcast_add(&w_exp)?;
    summed.max(1)
}
