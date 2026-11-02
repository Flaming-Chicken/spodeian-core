//! Topic 10: Floating-Point Arithmetic, Conditioning & Numerical Stability.
//!
//! Provides native Candle implementations for:
//! 1. Welford's single-pass online mean and population variance.
//! 2. Numerically stable Log-Sum-Exp (LSE) with max subtraction.
//! 3. Neumaier compensated summation.
//! 4. `StableLinear` forward projection and stable logit reduction.
//!
//! # Mathematical Invariants
//! - Welford's recurrence preserves precision: $\sigma^2 \ge 0$ guaranteed against $\mu \gg \sigma$ catastrophic cancellation.
//! - Stable LSE identity: $\operatorname{LSE}(\mathbf{x}) = m + \ln \sum_{i=1}^D \exp(x_i - m)$ where $m = \max_i x_i$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t10_floating_point_arithmetic_conditioning_and_numerical_stability::{
//!     log_sum_exp_stable, welford_mean_var,
//! };
//!
//! let device = Device::Cpu;
//! let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
//! let (mean, var) = welford_mean_var(&x, &device).unwrap();
//! assert_eq!(mean.dims(), &[2]);
//! assert_eq!(var.dims(), &[2]);
//! ```

use candle_core::{Device, Result, Tensor};

/// Single-pass numerically stable online mean and population variance via Welford's algorithm.
pub fn welford_mean_var(x: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
    let dims = x.dims();
    let n = dims[0];
    let d = dims[1];
    let x_flat = x.to_vec2::<f32>()?;

    let mut mean = vec![0.0f32; d];
    let mut m2 = vec![0.0f32; d];

    for i in 0..n {
        let count = (i + 1) as f32;
        for j in 0..d {
            let val = x_flat[i][j];
            let delta = val - mean[j];
            mean[j] += delta / count;
            let delta2 = val - mean[j];
            m2[j] += delta * delta2;
        }
    }

    let mut var = vec![0.0f32; d];
    for j in 0..d {
        var[j] = m2[j] / (n as f32);
    }

    let mean_t = Tensor::from_vec(mean, (d,), device)?;
    let var_t = Tensor::from_vec(var, (d,), device)?;
    Ok((mean_t, var_t))
}

/// Numerically stable Log-Sum-Exp along the last dimension of a 2D tensor.
pub fn log_sum_exp_stable(x: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = x.dims();
    let n = dims[0];
    let d = dims[1];
    let x_flat = x.to_vec2::<f32>()?;

    let mut lse_vec = vec![0.0f32; n];

    for i in 0..n {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..d {
            if x_flat[i][j] > max_val {
                max_val = x_flat[i][j];
            }
        }

        let mut sum_exp = 0.0f32;
        for j in 0..d {
            sum_exp += (x_flat[i][j] - max_val).exp();
        }

        lse_vec[i] = max_val + sum_exp.ln();
    }

    Tensor::from_vec(lse_vec, (n,), device)
}

/// Neumaier compensated summation along dimension 0 of a 2D tensor.
pub fn neumaier_sum_2d(x: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = x.dims();
    let n = dims[0];
    let d = dims[1];
    let x_flat = x.to_vec2::<f32>()?;

    let mut sum = vec![0.0f32; d];
    let mut comp = vec![0.0f32; d];

    for i in 0..n {
        for j in 0..d {
            let val = x_flat[i][j];
            let t = sum[j] + val;
            let c = if sum[j].abs() >= val.abs() {
                (sum[j] - t) + val
            } else {
                (val - t) + sum[j]
            };
            comp[j] += c;
            sum[j] = t;
        }
    }

    let mut result = vec![0.0f32; d];
    for j in 0..d {
        result[j] = sum[j] + comp[j];
    }

    Tensor::from_vec(result, (d,), device)
}

/// Linear projection module with numerically stable forward operations.
#[derive(Debug, Clone)]
pub struct StableLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl StableLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.matmul(&self.weight)?;
        logits.broadcast_add(&self.bias)
    }

    pub fn forward_lse(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let logits = self.forward(x)?;
        log_sum_exp_stable(&logits, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_variance() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 10.0], [3.0, 10.0], [5.0, 10.0]], &device).unwrap();
        let (mean, var) = welford_mean_var(&x, &device).unwrap();
        let mean_vec = mean.to_vec1::<f32>().unwrap();
        let var_vec = var.to_vec1::<f32>().unwrap();

        assert!((mean_vec[0] - 3.0).abs() < 1e-6);
        assert!((mean_vec[1] - 10.0).abs() < 1e-6);
        assert!((var_vec[0] - 8.0 / 3.0).abs() < 1e-6);
        assert!(var_vec[1].abs() < 1e-6);
    }

    #[test]
    fn test_log_sum_exp_finite() {
        let device = Device::Cpu;
        // Large values that would overflow naive exp(1000.0) -> inf
        let x = Tensor::new(&[[1000.0f32, 1000.0]], &device).unwrap();
        let lse = log_sum_exp_stable(&x, &device).unwrap();
        let lse_val = lse.to_vec1::<f32>().unwrap()[0];
        // ln(exp(1000) + exp(1000)) = 1000 + ln(2) = 1000.6931...
        assert!((lse_val - (1000.0 + 2.0f32.ln())).abs() < 1e-4);
        assert!(lse_val.is_finite());
    }
}
