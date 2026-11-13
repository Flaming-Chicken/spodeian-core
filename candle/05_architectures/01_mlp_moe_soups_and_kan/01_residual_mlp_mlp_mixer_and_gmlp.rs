//! Category 05 Topic 01: Residual MLP, MLP-Mixer, and gMLP.
//!
//! Provides native Candle implementations for:
//! 1. MLP-Mixer Block (Tolstikhin et al., 2021) with interleaved Token-mixing and Channel-mixing MLPs.
//! 2. gMLP Block (Liu et al., 2021) with Spatial Gating Unit (SGU) for cross-token interactions.
//! 3. ResMLP Block (Touvron et al., 2022) with affine channel scaling and LayerScale.
//!
//! # Mathematical Invariants
//! - Token vs Channel Factorization: Replaces $O(N^2)$ quadratic self-attention with independent
//!   $O(S \cdot D_{\text{token}})$ token-mixing and $O(C \cdot D_{\text{channel}})$ channel-mixing GEMMs.
//! - Spatial Gating Unit: $\text{SGU}(Z) = Z_1 \odot (W_{\text{sgu}} Z_2 + b_{\text{sgu}})$ where $b_{\text{sgu}} \approx 1$
//!   ensures identity initialization at step 0.

use candle_core::{D, Result, Tensor};

/// Linear projection helper for 3D tensors: reshapes [D0, D1, D2] -> [D0*D1, D2] -> GEMM -> reshape back.
pub fn linear_3d(x: &Tensor, weight_t: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (d0, d1, d2) = x.dims3()?;
    let x_2d = x.contiguous()?.reshape((d0 * d1, d2))?;
    let out_2d = x_2d.matmul(weight_t)?;
    let out_features = weight_t.dim(1)?;
    let out = out_2d.reshape((d0, d1, out_features))?;
    if let Some(b) = bias {
        out.broadcast_add(b)
    } else {
        Ok(out)
    }
}

/// Numerically stable LayerNorm along innermost channel dimension.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let diff = x.broadcast_sub(&mean)?;
    let sq_diff = diff.sqr()?;
    let var = sq_diff.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let norm = diff.broadcast_div(&std)?;
    let scaled = norm.broadcast_mul(weight)?;
    scaled.broadcast_add(bias)
}

/// Evaluates 1 complete MLP-Mixer block with Pre-LayerNorm and residual connections.
pub fn mlp_mixer_forward(
    x: &Tensor,
    w1_token: &Tensor,
    b1_token: &Tensor,
    w2_token: &Tensor,
    b2_token: &Tensor,
    w1_chan: &Tensor,
    b1_chan: &Tensor,
    w2_chan: &Tensor,
    b2_chan: &Tensor,
    ln1_w: &Tensor,
    ln1_b: &Tensor,
    ln2_w: &Tensor,
    ln2_b: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    // 1. Token-mixing: LayerNorm across channels, transpose to [B, C, S], Linear on tokens, transpose back
    let norm1 = layer_norm(x, ln1_w, ln1_b, eps)?;
    let x_t = norm1.transpose(1, 2)?;
    let w1_tok_t = w1_token.t()?;
    let h_tok = linear_3d(&x_t, &w1_tok_t, Some(b1_token))?.gelu_erf()?;
    let w2_tok_t = w2_token.t()?;
    let out_tok = linear_3d(&h_tok, &w2_tok_t, Some(b2_token))?.transpose(1, 2)?;
    let y = (x + out_tok)?;

    // 2. Channel-mixing: LayerNorm across channels, Linear on channels
    let norm2 = layer_norm(&y, ln2_w, ln2_b, eps)?;
    let w1_chan_t = w1_chan.t()?;
    let h_chan = linear_3d(&norm2, &w1_chan_t, Some(b1_chan))?.gelu_erf()?;
    let w2_chan_t = w2_chan.t()?;
    let out_chan = linear_3d(&h_chan, &w2_chan_t, Some(b2_chan))?;
    let z = (&y + out_chan)?;
    Ok(z)
}

/// Evaluates 1 Spatial Gating Unit (SGU): $Z_1 \odot (W_{\text{sgu}} Z_2 + b_{\text{sgu}})$.
pub fn spatial_gating_unit_forward(z: &Tensor, w_sgu: &Tensor, b_sgu: &Tensor) -> Result<Tensor> {
    let last_dim = z.dim(D::Minus1)?;
    let half_dim = last_dim / 2;
    let z1 = z.narrow(D::Minus1, 0, half_dim)?;
    let z2 = z.narrow(D::Minus1, half_dim, half_dim)?;

    // Linear projection across sequence dimension S: (w_sgu @ z2)^T = z2^T @ w_sgu^T
    let z2_t = z2.transpose(1, 2)?;
    let w_sgu_t = w_sgu.t()?;
    let gate_proj = linear_3d(&z2_t, &w_sgu_t, None)?.transpose(1, 2)?;
    let gate = gate_proj.broadcast_add(b_sgu)?;
    z1.mul(&gate)
}

/// Evaluates 1 gMLP block with Spatial Gating Unit and Pre-LayerNorm.
pub fn gmlp_forward(
    x: &Tensor,
    w_in: &Tensor,
    b_in: &Tensor,
    w_sgu: &Tensor,
    b_sgu: &Tensor,
    w_out: &Tensor,
    b_out: &Tensor,
    ln_w: &Tensor,
    ln_b: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let norm = layer_norm(x, ln_w, ln_b, eps)?;
    let w_in_t = w_in.t()?;
    let z = linear_3d(&norm, &w_in_t, Some(b_in))?.gelu_erf()?;
    let gated = spatial_gating_unit_forward(&z, w_sgu, b_sgu)?;
    let w_out_t = w_out.t()?;
    let out = linear_3d(&gated, &w_out_t, Some(b_out))?;
    x + out
}

/// Evaluates 1 ResMLP block with affine channel scaling and LayerScale.
pub fn resmlp_forward(
    x: &Tensor,
    scale_token: &Tensor,
    shift_token: &Tensor,
    w_token: &Tensor,
    b_token: &Tensor,
    ls_token: &Tensor,
    scale_chan: &Tensor,
    shift_chan: &Tensor,
    w1_chan: &Tensor,
    b1_chan: &Tensor,
    w2_chan: &Tensor,
    b2_chan: &Tensor,
    ls_chan: &Tensor,
) -> Result<Tensor> {
    // 1. Token mixing
    let aff1 = x.broadcast_mul(scale_token)?.broadcast_add(shift_token)?;
    let aff1_t = aff1.transpose(1, 2)?;
    let w_tok_t = w_token.t()?;
    let tok = linear_3d(&aff1_t, &w_tok_t, Some(b_token))?.transpose(1, 2)?;
    let y = (x + tok.broadcast_mul(ls_token)?)?;

    // 2. Channel mixing
    let aff2 = y.broadcast_mul(scale_chan)?.broadcast_add(shift_chan)?;
    let w1_c_t = w1_chan.t()?;
    let h_chan = linear_3d(&aff2, &w1_c_t, Some(b1_chan))?.gelu_erf()?;
    let w2_c_t = w2_chan.t()?;
    let chan = linear_3d(&h_chan, &w2_c_t, Some(b2_chan))?;
    let z = (&y + chan.broadcast_mul(ls_chan)?)?;
    Ok(z)
}

/// Linear anchor forward pass for transitive cross-track parity verification.
pub fn linear_anchor_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}
