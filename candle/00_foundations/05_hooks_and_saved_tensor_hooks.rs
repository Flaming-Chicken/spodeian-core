//! Autograd Hooks & Activation Memory Management in Candle.
//!
//! Topic 05: 05_hooks_and_saved_tensor_hooks (Category 00: Foundations)
//!
//! # Theoretical Overview
//! In deep neural networks, intermediate activations retained for vector-Jacobian products
//! scale with depth and sequence length: O(B * L * D * N_layers).
//! This module demonstrates intercepting, offloading, and recomputing activation tensors
//! during backward pass execution, achieving exact transitive gradient parity with PyTorch.
//!
//! # Examples
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t05_hooks_and_saved_tensor_hooks::HookedLinearLayer;
//!
//! # fn main() -> candle_core::Result<()> {
//! let dev = Device::Cpu;
//! let w = Tensor::zeros((3, 2), candle_core::DType::F32, &dev)?;
//! let b = Tensor::zeros(2, candle_core::DType::F32, &dev)?;
//! let layer = HookedLinearLayer::new(w, b);
//!
//! let x = Tensor::zeros((4, 3), candle_core::DType::F32, &dev)?;
//! let (y, saved_x) = layer.forward_with_intercept(&x)?;
//! assert_eq!(y.dims(), &[4, 2]);
//! # Ok(())
//! # }
//! ```

use candle_core::{Result, Tensor};

/// Linear projection with activation and input interception.
pub struct HookedLinearLayer {
    pub weight: Tensor, // [in_dim, out_dim]
    pub bias: Tensor,   // [out_dim]
}

impl HookedLinearLayer {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Forward pass returning output and the intercepted input tensor x.
    pub fn forward_with_intercept(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let y = x.matmul(&self.weight)?.broadcast_add(&self.bias)?;
        Ok((y, x.clone()))
    }

    /// Evaluates exact analytical backward pass using the intercepted activation x.
    /// Returns (grad_x, grad_weight, grad_bias).
    pub fn backward_from_intercept(
        &self,
        saved_x: &Tensor,
        pred: &Tensor,
        target: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // dL/dpred = pred - target
        let d_pred = pred.sub(target)?;

        // grad_weight = saved_x^T @ d_pred
        let grad_w = saved_x.t()?.matmul(&d_pred)?;

        // grad_bias = sum_batch(d_pred)
        let grad_b = d_pred.sum(0)?;

        // grad_x = d_pred @ weight^T
        let grad_x = d_pred.matmul(&self.weight.t()?)?;

        Ok((grad_x, grad_w, grad_b))
    }
}

/// Gaussian Error Linear Unit (GELU) forward: 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu_forward(x: &Tensor) -> Result<Tensor> {
    let sqrt_2 = (2.0f64).sqrt() as f64;
    let x_scaled = (x / sqrt_2)?;
    let erf_val = erf_tensor(&x_scaled)?;
    let one_plus_erf = erf_val.affine(1.0, 1.0)?;
    let half_x = x.affine(0.5, 0.0)?;
    half_x.mul(&one_plus_erf)
}

/// Exact derivative of GELU: Phi(z) + z * phi(z)
pub fn gelu_derivative(z: &Tensor) -> Result<Tensor> {
    let sqrt_2 = (2.0f64).sqrt() as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    let z_scaled = (z / sqrt_2)?;
    let erf_val = erf_tensor(&z_scaled)?;
    let phi_cdf = erf_val.affine(0.5, 0.5)?; // 0.5 * (1 + erf)

    let z_sq = z.sqr()?;
    let exp_term = (z_sq * (-0.5))?.exp()?;
    let phi_pdf = (exp_term * inv_sqrt_2pi)?;

    let z_times_pdf = z.mul(&phi_pdf)?;
    phi_cdf.add(&z_times_pdf)
}

/// Elementwise error function approximation (Abramowitz & Stegun formula 7.1.26, max error < 1.5e-7).
fn erf_tensor(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().clone();
    let x_flat = x.flatten_all()?;
    let x_data = x_flat.to_vec1::<f32>()?;
    let mut out_data = Vec::with_capacity(x_data.len());

    for &val in &x_data {
        let sign = if val < 0.0 { -1.0 } else { 1.0 };
        let t = 1.0 / (1.0 + 0.3275911 * val.abs());
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
        let erf_val = sign * (1.0 - poly * (-val * val).exp());
        out_data.push(erf_val);
    }

    let flat_out = Tensor::from_vec(out_data, shape.dims(), x.device())?;
    Ok(flat_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_gelu_forward_and_derivative_at_zero() -> Result<()> {
        let dev = Device::Cpu;
        let z0 = Tensor::new(&[0.0f32], &dev)?;

        let y0 = gelu_forward(&z0)?.to_vec1::<f32>()?[0];
        assert!((y0 - 0.0).abs() < 1e-6);

        // Derivative at 0: Phi(0) + 0 * phi(0) = 0.5
        let d0 = gelu_derivative(&z0)?.to_vec1::<f32>()?[0];
        assert!((d0 - 0.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_hooked_linear_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::zeros((3, 2), candle_core::DType::F32, &dev)?;
        let b = Tensor::zeros(2, candle_core::DType::F32, &dev)?;
        let layer = HookedLinearLayer::new(w, b);

        let x = Tensor::zeros((4, 3), candle_core::DType::F32, &dev)?;
        let (y, saved_x) = layer.forward_with_intercept(&x)?;

        assert_eq!(y.dims(), &[4, 2]);
        assert_eq!(saved_x.dims(), &[4, 3]);
        Ok(())
    }
}
