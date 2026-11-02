//! Torch Library Custom Ops and Fake Kernels
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Custom L1 proximal shrinkage operator:
/// shrink(x, lambd) = sign(x) * max(|x| - lambd, 0)
pub fn custom_soft_shrink(x: &Tensor, lambd: f32) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            let val = x_vec[i][j];
            let sign = if val > 0.0 {
                1.0f32
            } else if val < 0.0 {
                -1.0f32
            } else {
                0.0f32
            };
            let mag = (val.abs() - lambd).max(0.0f32);
            out[i][j] = sign * mag;
        }
    }
    Tensor::new(out, x.device())
}

/// Row-wise 8-bit asymmetric quantization: returns (x_hat, scale, zero_point)
pub fn custom_asymmetric_quantize(x: &Tensor, eps: f32) -> Result<(Tensor, Tensor, Tensor)> {
    let x_vec = x.to_vec2::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut x_hat = vec![vec![0.0f32; d]; b];
    let mut scales = vec![vec![0.0f32; 1]; b];
    let mut zeros = vec![vec![0.0f32; 1]; b];

    for i in 0..b {
        let min_val = x_vec[i].iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = x_vec[i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s = ((max_val - min_val) / 255.0f32).max(eps);
        let z = ((-min_val / s).round()).clamp(0.0f32, 255.0f32);

        scales[i][0] = s;
        zeros[i][0] = z;

        for j in 0..d {
            let q = ((x_vec[i][j] / s).round() + z).clamp(0.0f32, 255.0f32);
            x_hat[i][j] = (q - z) * s;
        }
    }

    let dev = x.device();
    Ok((
        Tensor::new(x_hat, dev)?,
        Tensor::new(scales, dev)?,
        Tensor::new(zeros, dev)?,
    ))
}

/// Depthwise 1D convolution with no group cross-talk.
pub fn custom_depthwise_conv1d(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec3::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let b_vec = bias.to_vec1::<f32>()?;

    let batch = x_vec.len();
    let channels = x_vec[0].len();
    let length = x_vec[0][0].len();
    let kernel_size = w_vec[0].len();
    let out_len = length - kernel_size + 1;

    let mut out = vec![vec![vec![0.0f32; out_len]; channels]; batch];
    for i in 0..batch {
        for c in 0..channels {
            for t in 0..out_len {
                let mut sum = b_vec[c];
                for k in 0..kernel_size {
                    sum += x_vec[i][c][t + k] * w_vec[c][k];
                }
                out[i][c][t] = sum;
            }
        }
    }
    Tensor::new(out, x.device())
}
