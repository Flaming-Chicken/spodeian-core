//! Block Floating Point BFP Microscaling MXFP and NVFP4
//! in candle/13_precision_and_numerics.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Elementwise minifloat quantizer with subnormal support.
pub fn quantize_minifloat_f32(v: f32, _e_bits: i32, m_bits: i32, bias: i32, max_val: f32) -> f32 {
    if v == 0.0 {
        return 0.0;
    }
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let abs_v = v.abs();
    let e = abs_v.log2().floor();
    let e_min_normal = 1.0 - (bias as f32);
    let step = if e < e_min_normal {
        (2.0f32).powf(e_min_normal - (m_bits as f32))
    } else {
        (2.0f32).powf(e - (m_bits as f32))
    };
    let q = (abs_v / step).round() * step;
    let q_clamped = q.clamp(0.0, max_val);
    sign * q_clamped
}

/// Block Floating Point (BFP) quantization with shared exponent and signed integer mantissa.
pub fn block_floating_point_quantize(
    x: &Tensor,
    block_size: usize,
    mantissa_bits: usize,
) -> Result<(Tensor, Tensor)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut dequant_vec = Vec::with_capacity(vec.len());
    let mut scales_vec = Vec::new();
    let max_int = ((1 << (mantissa_bits - 1)) - 1) as f32;

    for chunk in vec.chunks(block_size) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if amax == 0.0 {
            scales_vec.push(1.0f32);
            for _ in 0..chunk.len() {
                dequant_vec.push(0.0f32);
            }
        } else {
            let shared_exp = amax.log2().floor();
            let shared_scale = (2.0f32).powf(shared_exp);
            scales_vec.push(shared_scale);

            for &v in chunk {
                let norm = v / shared_scale;
                let q = (norm * max_int).round().clamp(-max_int, max_int);
                let deq = (q / max_int) * shared_scale;
                dequant_vec.push(deq);
            }
        }
    }

    let n_scales = scales_vec.len();
    let dequant = Tensor::from_vec(dequant_vec, x.shape(), x.device())?;
    let scales = Tensor::from_vec(scales_vec, &[n_scales], x.device())?;
    Ok((dequant, scales))
}

/// OCP Microscaling Specification (MXFP4):
/// Block size 32 with E8M0 power-of-2 shared scale factor and FP4 E2M1 elements (max 6.0).
pub fn ocp_mxfp4_quantize(x: &Tensor, block_size: usize) -> Result<(Tensor, Tensor)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut dequant_vec = Vec::with_capacity(vec.len());
    let mut scales_vec = Vec::new();

    for chunk in vec.chunks(block_size) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if amax == 0.0 {
            scales_vec.push(1.0f32);
            for _ in 0..chunk.len() {
                dequant_vec.push(0.0f32);
            }
        } else {
            let exp_scale = (amax / 6.0).log2().ceil();
            let scale_block = (2.0f32).powf(exp_scale);
            scales_vec.push(scale_block);

            for &v in chunk {
                let elem_scaled = v / scale_block;
                let elem_q = quantize_minifloat_f32(elem_scaled, 2, 1, 1, 6.0);
                dequant_vec.push(elem_q * scale_block);
            }
        }
    }

    let n_scales = scales_vec.len();
    let dequant = Tensor::from_vec(dequant_vec, x.shape(), x.device())?;
    let scales = Tensor::from_vec(scales_vec, &[n_scales], x.device())?;
    Ok((dequant, scales))
}

/// NVIDIA Blackwell-style two-level scaling:
/// Tensor-level global scale + Block-level local scale with FP4 E2M1 elements.
pub fn nvfp4_two_level_quantize(x: &Tensor, block_size: usize) -> Result<(Tensor, f64, Tensor)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let global_max = vec.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let global_scale = ((global_max / 6.0).max(1e-8)) as f64;
    let inv_glob = 1.0f32 / (global_scale as f32);

    let x_scaled_vec: Vec<f32> = vec.into_iter().map(|v| v * inv_glob).collect();
    let x_scaled = Tensor::from_vec(x_scaled_vec, x.shape(), x.device())?;

    let (dequant_local, local_scales) = ocp_mxfp4_quantize(&x_scaled, block_size)?;
    let dequant = dequant_local.affine(global_scale, 0.0)?;

    Ok((dequant, global_scale, local_scales))
}
