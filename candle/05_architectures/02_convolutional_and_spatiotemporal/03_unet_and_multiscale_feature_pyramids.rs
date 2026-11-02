//! U-Net & Multiscale Feature Pyramids (U-Net skip concatenation, BiFPN fast normalized fusion)
//! in candle/05_architectures/02_convolutional_and_spatiotemporal.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// 2x2 Max Pooling with stride 2.
pub fn max_pool2d_2x2(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let out_h = h / 2;
    let out_w = w / 2;
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0.0f32; b * c * out_h * out_w];

    for bi in 0..b {
        for ci in 0..c {
            let in_base = (bi * c + ci) * (h * w);
            let out_base = (bi * c + ci) * (out_h * out_w);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih = oh * 2;
                    let iw = ow * 2;
                    let v00 = x_vec[in_base + ih * w + iw];
                    let v01 = x_vec[in_base + ih * w + iw + 1];
                    let v10 = x_vec[in_base + (ih + 1) * w + iw];
                    let v11 = x_vec[in_base + (ih + 1) * w + iw + 1];
                    let max_v = v00.max(v01).max(v10).max(v11);
                    out[out_base + oh * out_w + ow] = max_v;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_h, out_w), x.device())
}

/// 2x Nearest Neighbor Upsampling.
pub fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let out_h = h * 2;
    let out_w = w * 2;
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0.0f32; b * c * out_h * out_w];

    for bi in 0..b {
        for ci in 0..c {
            let in_base = (bi * c + ci) * (h * w);
            let out_base = (bi * c + ci) * (out_h * out_w);
            for oh in 0..out_h {
                let ih = oh / 2;
                for ow in 0..out_w {
                    let iw = ow / 2;
                    out[out_base + oh * out_w + ow] = x_vec[in_base + ih * w + iw];
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_h, out_w), x.device())
}

/// Miniature U-Net Block forward pass.
pub fn unet_mini_forward(
    x: &Tensor,
    enc_w: &Tensor,
    enc_b: &Tensor,
    bot_w: &Tensor,
    bot_b: &Tensor,
    dec_w: &Tensor,
    dec_b: &Tensor,
) -> Result<Tensor> {
    // 1. Encoder Conv (4 -> 8, 3x3, pad 1) + ReLU
    let enc_b_exp = enc_b.reshape((1, enc_b.dims1()?, 1, 1))?;
    let e1 = x
        .conv2d(enc_w, 1, 1, 1, 1)?
        .broadcast_add(&enc_b_exp)?
        .relu()?;

    // 2. 2x2 Max Pooling
    let p1 = max_pool2d_2x2(&e1)?;

    // 3. Bottleneck Conv (8 -> 8, 3x3, pad 1) + ReLU
    let bot_b_exp = bot_b.reshape((1, bot_b.dims1()?, 1, 1))?;
    let b_feat = p1
        .conv2d(bot_w, 1, 1, 1, 1)?
        .broadcast_add(&bot_b_exp)?
        .relu()?;

    // 4. 2x Nearest Upsampling
    let up = upsample_nearest_2x(&b_feat)?;

    // 5. Skip Concatenation
    let cat_feat = Tensor::cat(&[&up, &e1], 1)?;

    // 6. Decoder Conv (16 -> 4, 3x3, pad 1) + ReLU
    let dec_b_exp = dec_b.reshape((1, dec_b.dims1()?, 1, 1))?;
    cat_feat
        .conv2d(dec_w, 1, 1, 1, 1)?
        .broadcast_add(&dec_b_exp)?
        .relu()
}

/// BiFPN Fast Normalized Feature Fusion forward pass.
pub fn bifpn_fusion(
    p1: &Tensor,
    p2: &Tensor,
    w: &Tensor,
    conv_w: &Tensor,
    conv_b: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let w0 = w_vec[0].max(0.0f32);
    let w1 = w_vec[1].max(0.0f32);
    let total_w = w0 + w1 + eps;
    let norm_w0 = (w0 / total_w) as f64;
    let norm_w1 = (w1 / total_w) as f64;

    let fused = ((p1 * norm_w0)? + (p2 * norm_w1)?)?;
    let conv_b_exp = conv_b.reshape((1, conv_b.dims1()?, 1, 1))?;
    fused.conv2d(conv_w, 0, 1, 1, 1)?.broadcast_add(&conv_b_exp)
}
