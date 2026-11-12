//! Automatic Mixed Precision Autocast and Gradscaler
//! in candle/13_precision_and_numerics.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Simulates PyTorch GradScaler dynamic loss scaling step.
pub fn dynamic_grad_scaler_step(
    grads: &Tensor,
    scale: f32,
    growth_tracker: usize,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
) -> Result<(Tensor, f32, usize, bool)> {
    let g_vec = grads.to_vec2::<f32>()?;
    let has_inf_nan = g_vec
        .iter()
        .flatten()
        .any(|&v| v.is_infinite() || v.is_nan());

    let unscaled = grads.affine((1.0f32 / scale) as f64, 0.0)?;

    if has_inf_nan {
        Ok((unscaled, scale * backoff_factor, 0, true))
    } else {
        let new_tracker = growth_tracker + 1;
        if new_tracker >= growth_interval {
            Ok((unscaled, scale * growth_factor, 0, false))
        } else {
            Ok((unscaled, scale, new_tracker, false))
        }
    }
}

/// Simulates AMP autocast policy: GEMM in FP16, Softmax in FP32.
pub fn amp_autocast_matmul_softmax(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let x_f16 = x.to_dtype(DType::F16)?;
    let w_f16 = w.to_dtype(DType::F16)?;
    let b_f16 = b.to_dtype(DType::F16)?;

    let z_f16 = x_f16.matmul(&w_f16)?.broadcast_add(&b_f16)?;
    let z_f32 = z_f16.to_dtype(DType::F32)?;
    let z_vec = z_f32.to_vec2::<f32>()?;

    let mut out = Vec::with_capacity(z_vec.len());
    for row in z_vec {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        out.push(probs);
    }
    Tensor::new(out, x.device())
}

/// Unscales gradients and applies norm-bounded clipping.
pub fn unscale_and_clip_grads(
    scaled_grads: &Tensor,
    scale: f32,
    max_norm: f32,
    eps: f32,
) -> Result<(Tensor, f32)> {
    let unscaled = scaled_grads.affine((1.0f32 / scale) as f64, 0.0)?;
    let u_vec = unscaled.to_vec2::<f32>()?;
    let norm_sq: f32 = u_vec.iter().flatten().map(|&v| v * v).sum();
    let total_norm = norm_sq.sqrt();
    let coeff = (max_norm / (total_norm + eps)).min(1.0f32);
    let clipped = unscaled.affine(coeff as f64, 0.0)?;
    Ok((clipped, total_norm))
}
