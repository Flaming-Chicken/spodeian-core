//! Specialized Convolutions & Channel Attention (CoordConv, Squeeze-and-Excitation, ECA-Net)
//! in candle/05_architectures/02_convolutional_and_spatiotemporal.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// CoordConv: Appends normalized spatial coordinate grids [-1, 1] before 1x1 convolution.
pub fn coord_conv2d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, _, h, w) = x.dims4()?;
    let dev = x.device();

    // Construct y coordinates in [-1, 1] of shape [1, 1, H, 1]
    let mut y_vec = Vec::with_capacity(h);
    if h == 1 {
        y_vec.push(0.0f32);
    } else {
        for i in 0..h {
            y_vec.push(-1.0f32 + 2.0f32 * (i as f32) / ((h - 1) as f32));
        }
    }
    let y_tensor = Tensor::from_vec(y_vec, (1, 1, h, 1), dev)?.broadcast_as((b, 1, h, w))?;

    // Construct x coordinates in [-1, 1] of shape [1, 1, 1, W]
    let mut x_vec = Vec::with_capacity(w);
    if w == 1 {
        x_vec.push(0.0f32);
    } else {
        for j in 0..w {
            x_vec.push(-1.0f32 + 2.0f32 * (j as f32) / ((w - 1) as f32));
        }
    }
    let x_tensor = Tensor::from_vec(x_vec, (1, 1, 1, w), dev)?.broadcast_as((b, 1, h, w))?;

    // Concatenate [x, x_coords, y_coords] along channel dim
    let x_with_coords = Tensor::cat(&[x, &x_tensor, &y_tensor], 1)?;

    // 1x1 Convolution: padding=0, stride=1, dilation=1, groups=1
    let out = x_with_coords.conv2d(weight, 0, 1, 1, 1)?;
    let bias_expanded = bias.reshape((1, bias.dims1()?, 1, 1))?;
    out.broadcast_add(&bias_expanded)
}

/// Squeeze-and-Excitation (SE) Block.
pub fn squeeze_and_excitation(
    x: &Tensor,
    fc1_w: &Tensor,
    fc1_b: &Tensor,
    fc2_w: &Tensor,
    fc2_b: &Tensor,
) -> Result<Tensor> {
    let (b, c, _, _) = x.dims4()?;

    // Global Average Pooling over spatial dimensions
    let gap = x.mean_keepdim(2)?.mean_keepdim(3)?.reshape((b, c))?;

    // MLP Bottleneck
    let fc1_out = gap.matmul(&fc1_w.t()?)?.broadcast_add(fc1_b)?;
    let hidden = fc1_out.relu()?;
    let fc2_out = hidden.matmul(&fc2_w.t()?)?.broadcast_add(fc2_b)?;
    let scale = candle_nn::ops::sigmoid(&fc2_out)?.reshape((b, c, 1, 1))?;

    x.broadcast_mul(&scale)
}

/// Efficient Channel Attention (ECA) Block.
pub fn efficient_channel_attention(x: &Tensor, conv_w: &Tensor, conv_b: &Tensor) -> Result<Tensor> {
    let (b, c, _, _) = x.dims4()?;

    // Global Average Pooling
    let gap = x.mean_keepdim(2)?.mean_keepdim(3)?.reshape((b, c))?;
    let w_vec = conv_w.flatten_all()?.to_vec1::<f32>()?;
    let b_val = conv_b.flatten_all()?.to_vec1::<f32>()?[0];

    // Padded 1D convolution with kernel size 3 and padding 1
    // Pad left and right with zeros
    let dev = x.device();
    let zero = Tensor::zeros((b, 1), candle_core::DType::F32, dev)?;
    let padded = Tensor::cat(&[&zero, &gap, &zero], 1)?;

    let left = padded.narrow(1, 0, c)?;
    let center = padded.narrow(1, 1, c)?;
    let right = padded.narrow(1, 2, c)?;

    let conv_out = (left * (w_vec[0] as f64))?
        .add(&(center * (w_vec[1] as f64))?)?
        .add(&(right * (w_vec[2] as f64))?)?
        .affine(1.0, b_val as f64)?;

    let scale = candle_nn::ops::sigmoid(&conv_out)?.reshape((b, c, 1, 1))?;
    x.broadcast_mul(&scale)
}
