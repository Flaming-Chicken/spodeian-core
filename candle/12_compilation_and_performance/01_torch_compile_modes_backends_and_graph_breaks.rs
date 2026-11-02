//! Torch Compile Modes Backends and Graph Breaks
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Fused SwiGLU activation (Shazeer 2020): y = (x * sigmoid(x)) * gate.
pub fn fused_swiglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let g_vec = gate.to_vec2::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            let val = x_vec[i][j];
            let sig = 1.0f32 / (1.0f32 + (-val).exp());
            let silu = val * sig;
            out[i][j] = silu * g_vec[i][j];
        }
    }
    Tensor::new(out, x.device())
}

/// Fused residual addition and Layer Normalization (Ba et al. 2016).
pub fn fused_add_layernorm(
    x: &Tensor,
    residual: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let r_vec = residual.to_vec2::<f32>()?;
    let gm = gamma.to_vec1::<f32>()?;
    let bt = beta.to_vec1::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        let mut sum = 0.0f32;
        let mut h = vec![0.0f32; d];
        for j in 0..d {
            h[j] = x_vec[i][j] + r_vec[i][j];
            sum += h[j];
        }
        let mean = sum / (d as f32);

        let mut var_sum = 0.0f32;
        for j in 0..d {
            let diff = h[j] - mean;
            var_sum += diff * diff;
        }
        let var = var_sum / (d as f32);
        let inv_std = 1.0f32 / (var + eps).sqrt();

        for j in 0..d {
            out[i][j] = ((h[j] - mean) * inv_std) * gm[j] + bt[j];
        }
    }
    Tensor::new(out, x.device())
}

/// Branchless conditional selection (torch.cond / jax.lax.cond equivalent).
pub fn tensor_cond_select(
    x: &Tensor,
    threshold: f32,
    scale_high: f32,
    scale_low: f32,
) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            let val = x_vec[i][j];
            out[i][j] = if val > threshold {
                val * scale_high
            } else {
                val * scale_low
            };
        }
    }
    Tensor::new(out, x.device())
}
