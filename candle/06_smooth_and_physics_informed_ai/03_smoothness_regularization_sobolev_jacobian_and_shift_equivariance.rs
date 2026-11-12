//! Smoothness Regularization: Sobolev Norms, Jacobian Penalties, and Shift Equivariance
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn erf_f32(x: f32) -> f32 {
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let p = 0.3275911f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();
    sign * y
}

fn gelu_derivative(u: &Tensor) -> Result<Tensor> {
    let u_vec = u.to_vec2::<f32>()?;
    let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;
    let inv_sqrt2pi = 1.0 / (2.0 * std::f32::consts::PI).sqrt();

    let d_vec: Vec<Vec<f32>> = u_vec
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|val| {
                    0.5 * (1.0 + erf_f32(val * inv_sqrt2))
                        + (val * inv_sqrt2pi) * (-0.5 * val * val).exp()
                })
                .collect()
        })
        .collect();
    Tensor::new(d_vec, u.device())
}

/// Evaluates exact analytical input gradient nabla_x y for scalar output y.
pub fn sobolev_gradient(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor) -> Result<Tensor> {
    let u = x.matmul(w1)?.broadcast_add(b1)?;
    let d_gelu = gelu_derivative(&u)?;
    let w2_t = w2.t()?;
    let weighted_d = d_gelu.broadcast_mul(&w2_t)?;
    weighted_d.matmul(&w1.t()?)
}

/// Evaluates squared Frobenius norm of the input-output Jacobian matrix.
pub fn jacobian_frobenius_sq(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor) -> Result<Tensor> {
    let u = x.matmul(w1)?.broadcast_add(b1)?;
    let d_gelu = gelu_derivative(&u)?;
    let d_out = w2.dims()[1];

    let mut sum_sq = Tensor::zeros((x.dims()[0], 1), DType::F32, x.device())?;
    for o in 0..d_out {
        let w2_col = w2.narrow(1, o, 1)?;
        let grad_o = d_gelu.broadcast_mul(&w2_col.t()?)?.matmul(&w1.t()?)?;
        let norm_sq = grad_o.sqr()?.sum_keepdim(1)?;
        sum_sq = (sum_sq + norm_sq)?;
    }
    sum_sq.squeeze(1)
}

/// Anti-aliased 1D BlurPool downsampling: conv1d with stride 2 and padding 1.
pub fn blur_downsample_1d(x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let b = dims[0];
    let c = dims[1];
    let l = dims[2];

    let k_vec = kernel.to_vec1::<f32>()?;
    let k0 = k_vec[0];
    let k1 = k_vec[1];
    let k2 = k_vec[2];

    let x_vec = x.to_vec3::<f32>()?;
    let out_len = l / 2;
    let mut out = vec![vec![vec![0.0f32; out_len]; c]; b];

    for bi in 0..b {
        for ci in 0..c {
            for step in 0..out_len {
                let idx_center = (step * 2) as isize;
                let val0 = if idx_center - 1 < 0 {
                    0.0
                } else {
                    x_vec[bi][ci][(idx_center - 1) as usize]
                };
                let val1 = x_vec[bi][ci][idx_center as usize];
                let val2 = if idx_center + 1 >= l as isize {
                    0.0
                } else {
                    x_vec[bi][ci][(idx_center + 1) as usize]
                };
                out[bi][ci][step] = val0 * k0 + val1 * k1 + val2 * k2;
            }
        }
    }
    Tensor::new(out, x.device())
}
