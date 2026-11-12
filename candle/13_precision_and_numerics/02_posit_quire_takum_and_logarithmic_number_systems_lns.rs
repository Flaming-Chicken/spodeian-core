//! Posit Quire Takum and Logarithmic Number Systems LNS
//! in candle/13_precision_and_numerics.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Simulates Posit<N, es> tapered precision arithmetic.
pub fn posit_tapered_quantize(x: &Tensor, n_bits: usize, es: usize) -> Result<Tensor> {
    let x_vec = x.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(x_vec.len());

    let useed = (1 << es) as f32;
    for v in x_vec {
        if v == 0.0f32 {
            out.push(0.0f32);
            continue;
        }
        let sign = if v < 0.0f32 { -1.0f32 } else { 1.0f32 };
        let abs_v = v.abs();
        let log2_x = abs_v.log2().floor();
        let k = (log2_x / useed).floor();
        let regime_bits = if k >= 0.0f32 { k + 2.0f32 } else { -k + 1.0f32 };
        let frac_bits = ((n_bits as f32) - 1.0f32 - regime_bits - (es as f32))
            .clamp(0.0f32, (n_bits - 2) as f32);
        let step = (2.0f32).powf(log2_x - frac_bits);
        let q = (abs_v / step).round() * step;
        out.push(sign * q);
    }
    Tensor::new(out, x.device())
}

/// Logarithmic Number System (LNS) multiplication.
pub fn lns_multiplication(a: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let a_vec = a.to_vec1::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(a_vec.len());

    for i in 0..a_vec.len() {
        let sign_out = a_vec[i].signum() * b_vec[i].signum();
        let log_a = (a_vec[i].abs() + eps).log2();
        let log_b = (b_vec[i].abs() + eps).log2();
        let log_out = log_a + log_b;
        out.push(sign_out * (2.0f32).powf(log_out));
    }
    Tensor::new(out, a.device())
}

/// Gaussian Log-Addition in LNS.
pub fn lns_gaussian_addition(a: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
    let a_vec = a.to_vec1::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(a_vec.len());

    for i in 0..a_vec.len() {
        let log_a = a_vec[i].max(eps).log2();
        let log_b = b_vec[i].max(eps).log2();
        let diff = (log_a - log_b).abs();
        let correction = (1.0f32 + (2.0f32).powf(-diff)).log2();
        let log_out = log_a.max(log_b) + correction;
        out.push((2.0f32).powf(log_out));
    }
    Tensor::new(out, a.device())
}

/// Kulisch Quire dot product accumulator simulation.
pub fn quire_dot_product_sim(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_mat = a.to_vec2::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let rows = a_mat.len();
    let cols = b_vec.len();

    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut sum = 0.0f64;
        for j in 0..cols {
            sum += (a_mat[i][j] as f64) * (b_vec[j] as f64);
        }
        out.push(sum as f32);
    }
    Tensor::new(out, a.device())
}
