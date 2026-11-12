//! LLM.int8() Mixed Precision Outlier Decomposition
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Per-row symmetric INT8 quantization.
pub fn quantize_row_int8(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, d) = x.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; b * d];
    let mut scales = vec![0.0f32; b];

    for i in 0..b {
        let mut max_abs = 0.0f32;
        for j in 0..d {
            let v = x_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[i] = s;
        for j in 0..d {
            let orig = x_vec[i][j];
            let q = (orig / s).round().clamp(-128.0, 127.0);
            deq_vec[i * d + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (b, d), x.device())?;
    let scale_tensor = Tensor::from_vec(scales, (b, 1), x.device())?;
    Ok((dequant, scale_tensor))
}

/// Per-column symmetric INT8 quantization.
pub fn quantize_col_int8(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = w.dims2()?;
    let w_vec = w.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; d_in * d_out];
    let mut scales = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let v = w_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[j] = s;
        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / s).round().clamp(-128.0, 127.0);
            deq_vec[i * d_out + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (d_in, d_out), w.device())?;
    let scale_tensor = Tensor::from_vec(scales, (1, d_out), w.device())?;
    Ok((dequant, scale_tensor))
}

/// LLM.int8() mixed-precision linear layer with outlier decomposition.
pub fn llm_int8_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    threshold: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (b, d_in) = x.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;

    let mut outlier_cols = Vec::new();
    let mut normal_cols = Vec::new();

    for j in 0..d_in {
        let mut col_max = 0.0f32;
        for i in 0..b {
            let v = x_vec[i][j].abs();
            if v > col_max {
                col_max = v;
            }
        }
        if col_max >= threshold {
            outlier_cols.push(j);
        } else {
            normal_cols.push(j);
        }
    }

    let k_outlier = outlier_cols.len();
    let k_normal = normal_cols.len();

    // FP path
    let y_fp = if k_outlier > 0 {
        let mut x_out_flat = vec![0.0f32; b * k_outlier];
        for i in 0..b {
            for (idx, &col) in outlier_cols.iter().enumerate() {
                x_out_flat[i * k_outlier + idx] = x_vec[i][col];
            }
        }
        let x_out_t = Tensor::from_vec(x_out_flat, (b, k_outlier), x.device())?;

        let mut w_out_flat = vec![0.0f32; k_outlier * d_out];
        for (idx, &row) in outlier_cols.iter().enumerate() {
            for j in 0..d_out {
                w_out_flat[idx * d_out + j] = w_vec[row][j];
            }
        }
        let w_out_t = Tensor::from_vec(w_out_flat, (k_outlier, d_out), weight.device())?;
        x_out_t.matmul(&w_out_t)?
    } else {
        Tensor::zeros((b, d_out), candle_core::DType::F32, x.device())?
    };

    // INT8 path
    let y_int8 = if k_normal > 0 {
        let mut x_norm_flat = vec![0.0f32; b * k_normal];
        for i in 0..b {
            for (idx, &col) in normal_cols.iter().enumerate() {
                x_norm_flat[i * k_normal + idx] = x_vec[i][col];
            }
        }
        let x_norm_t = Tensor::from_vec(x_norm_flat, (b, k_normal), x.device())?;

        let mut w_norm_flat = vec![0.0f32; k_normal * d_out];
        for (idx, &row) in normal_cols.iter().enumerate() {
            for j in 0..d_out {
                w_norm_flat[idx * d_out + j] = w_vec[row][j];
            }
        }
        let w_norm_t = Tensor::from_vec(w_norm_flat, (k_normal, d_out), weight.device())?;

        let (deq_x, _) = quantize_row_int8(&x_norm_t)?;
        let (deq_w, _) = quantize_col_int8(&w_norm_t)?;
        deq_x.matmul(&deq_w)?
    } else {
        Tensor::zeros((b, d_out), candle_core::DType::F32, x.device())?
    };

    let mut y_total = (&y_fp + &y_int8)?;
    if let Some(bi) = bias {
        y_total = y_total.broadcast_add(bi)?;
    }

    let outlier_f32: Vec<f32> = outlier_cols.iter().map(|&idx| idx as f32).collect();
    let outlier_tensor = Tensor::from_vec(outlier_f32, (k_outlier,), x.device())?;

    Ok((y_total, y_fp, y_int8, outlier_tensor))
}
