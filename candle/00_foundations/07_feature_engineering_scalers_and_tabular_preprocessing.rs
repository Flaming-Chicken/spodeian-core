//! Tabular Feature Engineering, Invertible Scalers & Serving Weight Folding in Candle.
//!
//! Topic 07: 07_feature_engineering_scalers_and_tabular_preprocessing (Category 00: Foundations)
//!
//! # Theoretical Overview
//! 1. Invertible Feature Standardization:
//!    Standardizes continuous columns with zero data leakage:
//!    $$z = \frac{x - \mu}{\sigma}, \quad x = z \cdot \sigma + \mu$$
//! 2. Parametric Monotonic Power Transforms:
//!    The Yeo-Johnson transform gaussianizes skewed tabular distributions across real domains:
//!    $$\psi(\lambda, x) = \begin{cases} \frac{(x + 1)^\lambda - 1}{\lambda} & x \ge 0, \lambda \ne 0 \\ \ln(x + 1) & x \ge 0, \lambda = 0 \\ -\frac{(-x + 1)^{2 - \lambda} - 1}{2 - \lambda} & x < 0, \lambda \ne 2 \\ -\ln(-x + 1) & x < 0, \lambda = 2 \end{cases}$$
//! 3. Serving Weight Folding:
//!    Fuses feature standardization statistics into linear layer weights for zero-overhead inference:
//!    $$W_{\text{fused}} = \text{diag}(\sigma^{-1}) W_{\text{head}}, \quad b_{\text{fused}} = b_{\text{head}} - (\mu / \sigma)^\top W_{\text{head}}$$
//!
//! # Examples
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t07_feature_engineering_scalers_and_tabular_preprocessing::{
//!     StandardScaler, FusedTabularLinear
//! };
//!
//! # fn main() -> candle_core::Result<()> {
//! let dev = Device::Cpu;
//! let mean = Tensor::new(&[10.0f32, 2.0], &dev)?;
//! let std = Tensor::new(&[2.0f32, 0.5], &dev)?;
//! let scaler = StandardScaler::new(mean, std);
//!
//! let x = Tensor::new(&[[12.0f32, 2.5]], &dev)?;
//! let z = scaler.transform(&x)?;
//! let x_rec = scaler.inverse_transform(&z)?;
//!
//! let diff = x.sub(&x_rec)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
//! assert!(diff < 1e-6);
//! # Ok(())
//! # }
//! ```

use candle_core::{Result, Tensor};

/// Invertible feature standardizer.
pub struct StandardScaler {
    pub mean: Tensor, // [dim]
    pub std: Tensor,  // [dim]
}

impl StandardScaler {
    pub fn new(mean: Tensor, std: Tensor) -> Self {
        Self { mean, std }
    }

    /// Transforms raw features into standardized coordinates: z = (x - mu) / sigma.
    pub fn transform(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
    }

    /// Reconstructs original raw features: x = z * sigma + mu.
    pub fn inverse_transform(&self, z: &Tensor) -> Result<Tensor> {
        z.broadcast_mul(&self.std)?.broadcast_add(&self.mean)
    }
}

/// Evaluates the Yeo-Johnson power transform elementwise.
pub fn yeo_johnson_transform(x: &Tensor, lmbda: f32) -> Result<Tensor> {
    let shape = x.shape();
    let flat_x = x.flatten_all()?.to_vec1::<f32>()?;
    let mut flat_out = Vec::with_capacity(flat_x.len());

    for val in flat_x {
        if val >= 0.0 {
            if (lmbda).abs() > 1e-6 {
                flat_out.push(((val + 1.0).powf(lmbda) - 1.0) / lmbda);
            } else {
                flat_out.push((val + 1.0).ln());
            }
        } else if (lmbda - 2.0).abs() > 1e-6 {
            flat_out.push(-((-val + 1.0).powf(2.0 - lmbda) - 1.0) / (2.0 - lmbda));
        } else {
            flat_out.push(-(-val + 1.0).ln());
        }
    }

    Tensor::from_vec(flat_out, shape, x.device())
}

/// Linear projection with preprocessing statistics folded into parameters.
pub struct FusedTabularLinear {
    pub weight: Tensor, // [in_dim, out_dim]
    pub bias: Tensor,   // [out_dim]
}

impl FusedTabularLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Folds mean/std standardization into linear head parameters:
    /// W_fused = diag(1/std) @ W_head
    /// b_fused = b_head - (mean / std) @ W_head
    pub fn from_scaled_head(
        mean: &Tensor,
        std: &Tensor,
        w_head: &Tensor,
        b_head: &Tensor,
    ) -> Result<Self> {
        let inv_std = std.recip()?;
        // Broadcast inv_std over w_head columns
        let inv_std_col = inv_std.reshape((inv_std.dims()[0], 1))?;
        let w_fused = w_head.broadcast_mul(&inv_std_col)?;

        // mean_over_std = mean / std
        let mean_over_std = mean.div(std)?;
        let mean_row = mean_over_std.reshape((1, mean_over_std.dims()[0]))?;
        let mean_proj = mean_row.matmul(w_head)?.reshape(b_head.dims())?;
        let b_fused = b_head.sub(&mean_proj)?;

        Ok(Self {
            weight: w_fused,
            bias: b_fused,
        })
    }

    /// Evaluates serving forward pass: y = x @ W_fused + b_fused.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_standard_scaler_roundtrip() -> Result<()> {
        let dev = Device::Cpu;
        let mean = Tensor::new(&[5.0f32, -2.0, 0.5], &dev)?;
        let std = Tensor::new(&[1.5f32, 3.0, 0.2], &dev)?;
        let scaler = StandardScaler::new(mean, std);

        let x = Tensor::new(&[[6.5f32, 1.0, 0.7], [3.5, -5.0, 0.3]], &dev)?;
        let z = scaler.transform(&x)?;
        let x_rec = scaler.inverse_transform(&z)?;

        let err = x
            .sub(&x_rec)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            err < 1e-6,
            "Scaler roundtrip error {:.2e} exceeds 1e-6",
            err
        );
        Ok(())
    }

    #[test]
    fn test_yeo_johnson_monotonic() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[-2.0f32, -0.5, 0.0, 1.5, 3.0], &dev)?;
        let yj = yeo_johnson_transform(&x, 0.5)?;
        let yj_vec = yj.to_vec1::<f32>()?;

        for i in 0..yj_vec.len() - 1 {
            assert!(
                yj_vec[i] < yj_vec[i + 1],
                "Yeo-Johnson must be strictly monotonic: {} >= {}",
                yj_vec[i],
                yj_vec[i + 1]
            );
        }
        Ok(())
    }
}
