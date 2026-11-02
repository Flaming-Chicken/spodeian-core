//! Diffusion Conditioning Controlnet and Ip Adapter
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Multiplies a 3D tensor [B, T, D_in] with a 2D weight [D_in, D_out] -> [B, T, D_out].
pub fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (d0, d1, d2) = x.dims3()?;
    let x_2d = x.contiguous()?.reshape((d0 * d1, d2))?;
    let out_2d = x_2d.matmul(w)?;
    out_2d.reshape((d0, d1, w.dim(1)?))
}

/// ControlNet residual block with zero-linear connections.
pub fn controlnet_residual_block(
    x: &Tensor,
    control_signal: &Tensor,
    backbone_w: &Tensor,
    control_w: &Tensor,
    z_in_w: &Tensor,
    z_in_b: &Tensor,
    z_out_w: &Tensor,
    z_out_b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let z_in = matmul_3d_2d(control_signal, z_in_w)?.broadcast_add(z_in_b)?;
    let ctrl_in = x.add(&z_in)?;
    let h_backbone = matmul_3d_2d(x, backbone_w)?;
    let h_control = matmul_3d_2d(&ctrl_in, control_w)?;
    let ctrl_out = matmul_3d_2d(&h_control, z_out_w)?.broadcast_add(z_out_b)?;
    let y = h_backbone.add(&ctrl_out)?;
    Ok((y, ctrl_out))
}

/// IP-Adapter decoupled cross-attention: Attn(Q, K_text, V_text) + lambda_img * Attn(Q, K_img, V_img).
pub fn ip_adapter_decoupled_cross_attention(
    q: &Tensor,
    k_text: &Tensor,
    v_text: &Tensor,
    k_img: &Tensor,
    v_img: &Tensor,
    scale: f64,
    lambda_img: f64,
) -> Result<Tensor> {
    // 1. Text cross-attention
    let scores_text = (q.matmul(&k_text.transpose(1, 2)?)? * scale)?;
    let attn_weights_text = candle_nn::ops::softmax(&scores_text, 2)?;
    let attn_text = attn_weights_text.matmul(v_text)?;

    // 2. Image cross-attention
    let scores_img = (q.matmul(&k_img.transpose(1, 2)?)? * scale)?;
    let attn_weights_img = candle_nn::ops::softmax(&scores_img, 2)?;
    let attn_img = attn_weights_img.matmul(v_img)?;

    // 3. Decoupled combination
    let scaled_img = (attn_img * lambda_img)?;
    attn_text.add(&scaled_img)
}
