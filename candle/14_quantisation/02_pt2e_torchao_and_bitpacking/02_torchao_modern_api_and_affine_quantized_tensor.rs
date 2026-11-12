//! TorchAO Modern API and Affine Quantized Tensor
//! in candle/14_quantisation/02_pt2e_torchao_and_bitpacking.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Affine Quantized Tensor (AQT) block-wise symmetric quantization.
pub fn affine_quantize_blockwise(
    weight: &Tensor,
    group_size: usize,
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let flat = weight.flatten_all()?.to_vec1::<f32>()?;
    let mut scales = Vec::new();
    let mut q_int = Vec::with_capacity(flat.len());
    let mut dequant = Vec::with_capacity(flat.len());

    for chunk in flat.chunks(group_size) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if amax > 0.0 { amax / qmax } else { 1.0f32 };
        let inv_scale = 1.0f32 / scale;
        scales.push(scale);

        for &v in chunk {
            let q = (v * inv_scale).round().clamp(qmin, qmax);
            q_int.push(q);
            dequant.push(q * scale);
        }
    }

    let n_scales = scales.len();
    let dequant_tensor = Tensor::from_vec(dequant, weight.shape(), weight.device())?;
    let scales_tensor = Tensor::from_vec(scales, &[n_scales], weight.device())?;
    let q_int_tensor = Tensor::from_vec(q_int, weight.shape(), weight.device())?;

    Ok((dequant_tensor, scales_tensor, q_int_tensor))
}

/// Dynamic Activation INT8, Static Weight INT8 (W8A8) GEMM.
pub fn dynamic_w8a8_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let x_dims = x.dims();
    let b = x_dims[0];
    let d_in = x_dims[1];
    let d_out = weight.dim(1)?;

    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
    let mut s_x_vec = Vec::with_capacity(b);
    let mut x_q_vec = Vec::with_capacity(b * d_in);

    for chunk in x_flat.chunks(d_in) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0f32 } else { 1.0f32 };
        let inv_s = 1.0f32 / s;
        s_x_vec.push(s);
        for &v in chunk {
            x_q_vec.push((v * inv_s).round().clamp(-128.0, 127.0));
        }
    }

    let w_flat = weight.flatten_all()?.to_vec1::<f32>()?;
    let mut s_w_vec = Vec::with_capacity(d_out);
    for col in 0..d_out {
        let mut col_amax = 0.0f32;
        for row in 0..d_in {
            let v = w_flat[row * d_out + col];
            col_amax = col_amax.max(v.abs());
        }
        let s = if col_amax > 0.0 {
            col_amax / 127.0f32
        } else {
            1.0f32
        };
        s_w_vec.push(s);
    }

    let mut w_q_vec = Vec::with_capacity(d_in * d_out);
    for row in 0..d_in {
        for col in 0..d_out {
            let v = w_flat[row * d_out + col];
            let inv_s = 1.0f32 / s_w_vec[col];
            w_q_vec.push((v * inv_s).round().clamp(-128.0, 127.0));
        }
    }

    let x_q = Tensor::from_vec(x_q_vec, (b, d_in), x.device())?;
    let w_q = Tensor::from_vec(w_q_vec, (d_in, d_out), weight.device())?;
    let y_int = x_q.matmul(&w_q)?;

    let s_x = Tensor::from_vec(s_x_vec, (b, 1), x.device())?;
    let s_w = Tensor::from_vec(s_w_vec, (1, d_out), weight.device())?;
    let scale_2d = s_x.broadcast_mul(&s_w)?;

    let y = y_int.mul(&scale_2d)?.broadcast_add(bias)?;
    Ok((y, s_x, s_w))
}

/// INT4 Weight-Only (W4A16) block-wise quantization with FP32/FP16 activation.
pub fn int4_weight_only_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    group_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (w_deq, scales, _) = affine_quantize_blockwise(weight, group_size, -8.0, 7.0)?;
    let y = x.matmul(&w_deq)?.broadcast_add(bias)?;
    Ok((y, w_deq, scales))
}
