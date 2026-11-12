//! Category 05 Subfolder 02 Topic 01: Classic to Modern CNNs: ResNet v1/v2, DenseNet, MobileNetV2, and ConvNeXt-V2 (GRN).
//!
//! Provides native Candle implementations for:
//! 1. Global Response Normalization (GRN) for ConvNeXt-V2.
//! 2. ConvNeXt-V2 Inverted Bottleneck Block with 7x7 Depthwise Convolutions.
//! 3. Pre-Activation ResNet Block (ResNet-v2).

use candle_core::{Result, Tensor};

/// Applies 2D convolution with broadcast bias addition.
pub fn conv2d_with_bias(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    padding: usize,
    groups: usize,
) -> Result<Tensor> {
    let out = x.conv2d(weight, padding, 1, 1, groups)?;
    let c_out = bias.dim(0)?;
    let b_4d = bias.reshape((1, c_out, 1, 1))?;
    out.broadcast_add(&b_4d)
}

/// Global Response Normalization (GRN) for ConvNeXt-V2:
/// G_c = sqrt(sum_{h, w} X_{c, h, w}^2 + eps)
/// N_c = G_c / (mean_j G_j + eps)
/// X' = gamma * (X * N) + beta + X
pub fn global_response_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let sum_spatial = sq.sum_keepdim(2)?.sum_keepdim(3)?;
    let gx = (sum_spatial + eps)?.sqrt()?;

    let mean_gx = gx.mean_keepdim(1)?;
    let nx = gx.broadcast_div(&(mean_gx + eps)?)?;

    let c = gamma.dim(0)?;
    let gamma_4d = gamma.reshape((1, c, 1, 1))?;
    let beta_4d = beta.reshape((1, c, 1, 1))?;

    let x_calibrated = x.broadcast_mul(&nx)?;
    let x_scaled = x_calibrated
        .broadcast_mul(&gamma_4d)?
        .broadcast_add(&beta_4d)?;
    x_scaled.add(x)
}

/// ConvNeXt-V2 Inverted Bottleneck Block forward pass.
pub fn convnext_v2_forward(
    x: &Tensor,
    dw_w: &Tensor,
    dw_b: &Tensor,
    pw1_w: &Tensor,
    pw1_b: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    pw2_w: &Tensor,
    pw2_b: &Tensor,
) -> Result<Tensor> {
    let in_channels = x.dim(1)?;
    let h_dw = conv2d_with_bias(x, dw_w, dw_b, 3, in_channels)?;
    let h_pw1 = conv2d_with_bias(&h_dw, pw1_w, pw1_b, 0, 1)?;
    let h_act = h_pw1.gelu_erf()?;
    let h_grn = global_response_norm(&h_act, gamma, beta, 1e-6)?;
    let h_pw2 = conv2d_with_bias(&h_grn, pw2_w, pw2_b, 0, 1)?;
    x.add(&h_pw2)
}

/// Pre-Activation ResNet Block forward pass (ReLU -> Conv -> ReLU -> Conv -> (+)).
pub fn preact_resnet_forward(
    x: &Tensor,
    res_w1: &Tensor,
    res_b1: &Tensor,
    res_w2: &Tensor,
    res_b2: &Tensor,
) -> Result<Tensor> {
    let r1 = x.relu()?;
    let r2 = conv2d_with_bias(&r1, res_w1, res_b1, 1, 1)?;
    let r3 = r2.relu()?;
    let r4 = conv2d_with_bias(&r3, res_w2, res_b2, 1, 1)?;
    x.add(&r4)
}
