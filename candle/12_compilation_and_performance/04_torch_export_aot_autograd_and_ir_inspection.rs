//! Torch Export AOT Autograd and IR Inspection
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Decomposed Core ATen Softmax operator.
pub fn aten_core_softmax(x: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let mut out = Vec::with_capacity(x_vec.len());
    for row in x_vec {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let sm_row: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();
        out.push(sm_row);
    }
    Tensor::new(out, x.device())
}

/// Decomposed Core ATen LayerNorm operator.
pub fn aten_core_layernorm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let gm = gamma.to_vec1::<f32>()?;
    let bt = beta.to_vec1::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        let sum: f32 = x_vec[i].iter().sum();
        let mean = sum / (d as f32);
        let var_sum: f32 = x_vec[i].iter().map(|&v| (v - mean) * (v - mean)).sum();
        let var = var_sum / (d as f32);
        let rsqrt = 1.0f32 / (var + eps).sqrt();
        for j in 0..d {
            out[i][j] = (x_vec[i][j] - mean) * rsqrt * gm[j] + bt[j];
        }
    }
    Tensor::new(out, x.device())
}

/// High-precision error function approximation (Abramowitz & Stegun 7.1.26, max error < 1.5e-7).
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911f32;
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let t = 1.0f32 / (1.0f32 + p * x_abs);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0f32 - poly * (-x_abs * x_abs).exp();
    sign * y
}

/// AOTAutograd captured joint forward and reverse-mode VJP adjoint graph:
/// Forward: z = x @ w + b; a = 0.5 * z * (1 + erf(z / sqrt(2)))
/// Backward: VJP adjoints for grad_w, grad_b, and grad_x
pub fn aot_forward_backward_affine_gelu(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    grad_out: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = w.to_vec2::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let gz_in = grad_out.to_vec2::<f32>()?;

    let batch = x_vec.len();
    let d_in = w_vec.len();
    let d_out = w_vec[0].len();

    let sqrt_2 = std::f32::consts::SQRT_2;
    let sqrt_2_pi = 2.5066282746310002f32;

    let mut a_out = vec![vec![0.0f32; d_out]; batch];
    let mut gz = vec![vec![0.0f32; d_out]; batch];

    // Compute forward activation and backward adjoint gz
    for i in 0..batch {
        for j in 0..d_out {
            let mut z = b_vec[j];
            for k in 0..d_in {
                z += x_vec[i][k] * w_vec[k][j];
            }
            let erf_val = erf_f32(z / sqrt_2);
            a_out[i][j] = 0.5f32 * z * (1.0f32 + erf_val);

            let d_gelu_dz = 0.5f32 * (1.0f32 + erf_val) + (z / sqrt_2_pi) * (-0.5f32 * z * z).exp();
            gz[i][j] = gz_in[i][j] * d_gelu_dz;
        }
    }

    // grad_w = x^T @ gz
    let mut grad_w = vec![vec![0.0f32; d_out]; d_in];
    for k in 0..d_in {
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for i in 0..batch {
                sum += x_vec[i][k] * gz[i][j];
            }
            grad_w[k][j] = sum;
        }
    }

    // grad_b = sum_i gz[i]
    let mut grad_b = vec![0.0f32; d_out];
    for j in 0..d_out {
        let mut sum = 0.0f32;
        for i in 0..batch {
            sum += gz[i][j];
        }
        grad_b[j] = sum;
    }

    // grad_x = gz @ w^T
    let mut grad_x = vec![vec![0.0f32; d_in]; batch];
    for i in 0..batch {
        for k in 0..d_in {
            let mut sum = 0.0f32;
            for j in 0..d_out {
                sum += gz[i][j] * w_vec[k][j];
            }
            grad_x[i][k] = sum;
        }
    }

    let dev = x.device();
    Ok((
        Tensor::new(a_out, dev)?,
        Tensor::new(grad_w, dev)?,
        Tensor::new(grad_b, dev)?,
        Tensor::new(grad_x, dev)?,
    ))
}
