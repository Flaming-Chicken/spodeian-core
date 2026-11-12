//! Topic 01: Linear Neuron and Invertible Transforms (Native Rust Candle Track).
//!
//! Provides the native Rust Candle implementation of the fundamental affine transform:
//!     y = x @ W + b
//! alongside its exact mathematical inverse:
//!     x = (y - b) @ W^{-1}
//! and foundational invertible coordinate normalization transforms.

use candle_core::{Result, Tensor};

/// Linear affine projection layer with forward and inverse passes.
///
/// # Example
/// ```rust
/// use candle_core::{Device, Tensor};
/// use ai_candle::c00_foundations::t01_linear_neuron_and_invertible_transforms::LinearNeuron;
///
/// let device = Device::Cpu;
/// let weight = Tensor::new(&[[2.0f32]], &device).unwrap();
/// let bias = Tensor::new(&[0.5f32], &device).unwrap();
/// let neuron = LinearNeuron::new(weight, bias);
/// let x = Tensor::new(&[[1.0f32]], &device).unwrap();
/// let y = neuron.forward(&x).unwrap();
/// assert_eq!(y.to_vec2::<f32>().unwrap(), vec![vec![2.5f32]]);
/// ```
pub struct LinearNeuron {
    pub weight: Tensor, // [in_dim, out_dim]
    pub bias: Tensor,   // [out_dim]
}

impl LinearNeuron {
    /// Constructs a new LinearNeuron from weight and bias tensors.
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Evaluates linear affine forward pass: y = x @ W + b
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let proj = x.matmul(&self.weight)?;
        proj.broadcast_add(&self.bias)
    }

    /// Evaluates exact inverse pass for 1D / scalar affine transform: x = (y - b) / w
    pub fn inverse_1d(&self, y: &Tensor) -> Result<Tensor> {
        let centered = y.broadcast_sub(&self.bias)?;
        centered.broadcast_div(&self.weight)
    }

    /// Evaluates affine inverse via weight matrix inversion: x = (y - b) @ W^{-1}
    pub fn inverse(&self, y: &Tensor, inv_weight: &Tensor) -> Result<Tensor> {
        let centered = y.broadcast_sub(&self.bias)?;
        centered.matmul(inv_weight)
    }
}

/// Invertible coordinate transformations
pub struct CoordinateTransforms;

impl CoordinateTransforms {
    /// Evaluates Z-score standardization: (x - mu) / sigma
    pub fn z_score(x: &Tensor, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let mean = x.mean_keepdim(0)?;
        // Population standard deviation
        let diff = x.broadcast_sub(&mean)?;
        let sq_diff = diff.sqr()?;
        let var = sq_diff.mean_keepdim(0)?;
        let std = (var + eps)?.sqrt()?;
        let normed = diff.broadcast_div(&std)?;
        Ok((normed, mean, std))
    }

    /// Inverts Z-score standardization: t * sigma + mu
    pub fn undo_z_score(t: &Tensor, mean: &Tensor, std: &Tensor) -> Result<Tensor> {
        let scaled = t.broadcast_mul(std)?;
        scaled.broadcast_add(mean)
    }

    /// Evaluates Min-Max scaling to [-1, 1]
    pub fn min_max(x: &Tensor, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let min = x.min_keepdim(0)?;
        let max = x.max_keepdim(0)?;
        let diff = max.broadcast_sub(&min)?;
        let diff_clamped = diff.clamp(eps, f64::MAX)?;

        let zero_to_one = x.broadcast_sub(&min)?.broadcast_div(&diff_clamped)?;
        let minus_one_to_one = ((zero_to_one * 2.0)? - 1.0)?;
        Ok((minus_one_to_one, min, diff_clamped))
    }

    /// Inverts Min-Max scaling
    pub fn undo_min_max(t: &Tensor, min: &Tensor, diff: &Tensor) -> Result<Tensor> {
        let zero_to_one = ((t + 1.0)? * 0.5)?;
        let scaled = zero_to_one.broadcast_mul(diff)?;
        scaled.broadcast_add(min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_affine_forward_and_inverse_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        // W = [[2.0]], b = [0.5]
        let weight = Tensor::new(&[[2.0f32]], &device)?;
        let bias = Tensor::new(&[0.5f32], &device)?;
        let neuron = LinearNeuron::new(weight, bias);

        let x = Tensor::new(&[[-1.0f32], [0.0f32], [1.5f32], [3.0f32]], &device)?;
        let y = neuron.forward(&x)?;
        let x_rec = neuron.inverse_1d(&y)?;

        let diff = (x.sub(&x_rec))?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "Reconstruction error {} exceeds 1e-5", diff);
        Ok(())
    }

    #[test]
    fn test_z_score_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[10.0f32], [20.0f32], [30.0f32], [40.0f32]], &device)?;
        let (normed, mean, std) = CoordinateTransforms::z_score(&x, 1e-7)?;
        let x_rec = CoordinateTransforms::undo_z_score(&normed, &mean, &std)?;

        let diff = (x.sub(&x_rec))?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "Z-Score reconstruction error {} exceeds 1e-5",
            diff
        );
        Ok(())
    }
}
