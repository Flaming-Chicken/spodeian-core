//! Topic 03: Autograd Mechanics, Complex Wirtinger & Custom Functions (Native Rust Candle Track).
//!
//! Demonstrates reverse-mode Vector-Jacobian Product (VJP) mechanics,
//! analytical gradient calculations, and complex Wirtinger derivatives.

use candle_core::{Result, Tensor};

/// Analytical Vector-Jacobian Product (VJP) evaluator for linear projections.
///
/// # Example
/// ```rust
/// use candle_core::{Device, Tensor};
/// use ai_candle::c00_foundations::t03_autograd_mechanics_complex_wirtinger_and_custom_functions::LinearVjp;
///
/// let device = Device::Cpu;
/// let x = Tensor::new(&[[1.0f32, 2.0f32]], &device).unwrap();
/// let w = Tensor::new(&[[0.5f32], [1.5f32]], &device).unwrap();
/// let b = Tensor::new(&[0.1f32], &device).unwrap();
/// let y = LinearVjp::forward(&x, &w, &b).unwrap();
/// assert_eq!(y.to_vec2::<f32>().unwrap(), vec![vec![3.6f32]]);
/// ```
pub struct LinearVjp;

impl LinearVjp {
    /// Evaluates forward pass: Y = X @ W + b
    pub fn forward(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let proj = x.matmul(w)?;
        proj.broadcast_add(b)
    }

    /// Evaluates analytical VJP given upstream gradient G:
    /// dX = G @ W^T
    /// dW = X^T @ G
    /// db = sum_batch(G)
    pub fn backward(g: &Tensor, x: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let w_t = w.t()?;
        let dx = g.matmul(&w_t)?;

        let x_t = x.t()?;
        let dw = x_t.matmul(g)?;

        let db = g.sum(0)?;

        Ok((dx, dw, db))
    }

    /// Evaluates complex Wirtinger conjugate gradient 2 * dL/dz* = (z - y)
    pub fn complex_wirtinger_grad(z: &Tensor, y: &Tensor) -> Result<Tensor> {
        z.sub(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_analytical_linear_vjp() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let w = Tensor::new(&[[0.5f32, -0.5], [1.0, 0.0]], &device)?;
        let b = Tensor::new(&[0.1f32, -0.1], &device)?;

        let _y = LinearVjp::forward(&x, &w, &b)?;
        let g = Tensor::new(&[[1.0f32, 1.0], [1.0, 1.0]], &device)?;

        let (dx, dw, db) = LinearVjp::backward(&g, &x, &w)?;

        assert_eq!(dx.dims(), &[2, 2]);
        assert_eq!(dw.dims(), &[2, 2]);
        assert_eq!(db.dims(), &[2]);
        Ok(())
    }

    #[test]
    fn test_wirtinger_grad_subtraction() -> Result<()> {
        let device = Device::Cpu;
        let z = Tensor::new(&[2.0f32, 3.0], &device)?;
        let y = Tensor::new(&[1.0f32, 1.5], &device)?;
        let grad = LinearVjp::complex_wirtinger_grad(&z, &y)?;

        let val = grad.to_vec1::<f32>()?;
        assert_eq!(val, vec![1.0, 1.5]);
        Ok(())
    }
}
