//! Floating Point Anatomy Subnormal Traps and Stochastic Logic
//! in candle/13_precision_and_numerics.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Simulates hardware Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ).
pub fn ftz_daz_simulate(x: &Tensor, threshold: f32) -> Result<Tensor> {
    let x_vec = x.to_vec1::<f32>()?;
    let out: Vec<f32> = x_vec
        .into_iter()
        .map(|v| if v.abs() < threshold { 0.0f32 } else { v })
        .collect();
    Tensor::new(out, x.device())
}

/// Simulates round-to-nearest-even conversion from FP32 to BF16.
pub fn fp32_to_bf16_rtne(x: &Tensor) -> Result<Tensor> {
    x.to_dtype(DType::BF16)?.to_dtype(DType::F32)
}

/// Stochastic computing unipolar multiplication via bitwise AND.
pub fn stochastic_unipolar_mul(
    p1: &Tensor,
    p2: &Tensor,
    u1: &Tensor,
    u2: &Tensor,
) -> Result<Tensor> {
    let p1_vec = p1.to_vec1::<f32>()?;
    let p2_vec = p2.to_vec1::<f32>()?;
    let u1_vec = u1.to_vec2::<f32>()?;
    let u2_vec = u2.to_vec2::<f32>()?;

    let batch = p1_vec.len();
    let length = u1_vec[0].len();
    let mut out = Vec::with_capacity(batch);

    for i in 0..batch {
        let p1_val = p1_vec[i];
        let p2_val = p2_vec[i];
        let mut count = 0.0f32;
        for k in 0..length {
            let b1 = if u1_vec[i][k] < p1_val {
                1.0f32
            } else {
                0.0f32
            };
            let b2 = if u2_vec[i][k] < p2_val {
                1.0f32
            } else {
                0.0f32
            };
            count += b1 * b2;
        }
        out.push(count / (length as f32));
    }
    Tensor::new(out, p1.device())
}

/// Stochastic computing bipolar value multiplication via bitwise XNOR.
pub fn stochastic_bipolar_mul(
    x1: &Tensor,
    x2: &Tensor,
    u1: &Tensor,
    u2: &Tensor,
) -> Result<Tensor> {
    let x1_vec = x1.to_vec1::<f32>()?;
    let x2_vec = x2.to_vec1::<f32>()?;
    let u1_vec = u1.to_vec2::<f32>()?;
    let u2_vec = u2.to_vec2::<f32>()?;

    let batch = x1_vec.len();
    let length = u1_vec[0].len();
    let mut out = Vec::with_capacity(batch);

    for i in 0..batch {
        let p1 = (x1_vec[i] + 1.0f32) * 0.5f32;
        let p2 = (x2_vec[i] + 1.0f32) * 0.5f32;
        let mut count = 0.0f32;
        for k in 0..length {
            let b1 = if u1_vec[i][k] < p1 { 1.0f32 } else { 0.0f32 };
            let b2 = if u2_vec[i][k] < p2 { 1.0f32 } else { 0.0f32 };
            let b_xnor = b1 * b2 + (1.0f32 - b1) * (1.0f32 - b2);
            count += b_xnor;
        }
        let p_out = count / (length as f32);
        out.push(2.0f32 * p_out - 1.0f32);
    }
    Tensor::new(out, x1.device())
}
