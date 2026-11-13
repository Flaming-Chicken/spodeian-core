//! Topic 04: Forward AD and Functional Transforms (Native Rust Candle Track).
//!
//! Demonstrates forward-mode directional derivatives (Jacobian-Vector Products, JVP),
//! dual numbers arithmetic, and vector-Jacobian products (VJP) in pure Rust with Candle.

use candle_core::{Result, Tensor};

/// Evaluator for Forward-Mode AD (JVP: J @ v) and Dual Number mechanics.
///
/// # Example
/// ```rust
/// use candle_core::{Device, Tensor};
/// use sml_candle::c00_foundations::t04_forward_ad_and_functional_transforms_torch_func::ForwardAdJvp;
///
/// let device = Device::Cpu;
/// let x = Tensor::new(&[[1.0f32, 2.0f32]], &device).unwrap();
/// let v = Tensor::new(&[[0.1f32, 0.2f32]], &device).unwrap();
/// let w = Tensor::new(&[[2.0f32], [3.0f32]], &device).unwrap();
/// let b = Tensor::new(&[0.5f32], &device).unwrap();
///
/// let (y, jvp) = ForwardAdJvp::linear_jvp(&x, &v, &w, &b).unwrap();
/// assert_eq!(y.to_vec2::<f32>().unwrap(), vec![vec![8.5f32]]);
/// assert_eq!(jvp.to_vec2::<f32>().unwrap(), vec![vec![0.8f32]]);
/// ```
pub struct ForwardAdJvp;

impl ForwardAdJvp {
    /// Evaluates linear forward pass along with forward-mode directional derivative (JVP):
    /// Primal: Y = X @ W + b
    /// Tangent: dY = V @ W
    pub fn linear_jvp(x: &Tensor, v: &Tensor, w: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
        let proj = x.matmul(w)?;
        let y = proj.broadcast_add(b)?;

        // Forward tangent propagation: J @ v = v @ W
        let jvp = v.matmul(w)?;

        Ok((y, jvp))
    }

    /// Computes per-sample gradients (dL_i / dW) across a batch:
    /// For MSE loss 0.5 * ||Y_i - Y_target_i||^2, dL_i / dW = X_i^T @ (Y_i - Y_target_i)
    pub fn per_sample_weights_grad(x: &Tensor, pred: &Tensor, target: &Tensor) -> Result<Tensor> {
        let diff = pred.sub(target)?;
        let (batch_size, in_dim) = x.dims2()?;
        let (_, out_dim) = pred.dims2()?;

        // Reshape x to [B, in_dim, 1] and diff to [B, 1, out_dim]
        let x_b = x.reshape((batch_size, in_dim, 1))?;
        let diff_b = diff.reshape((batch_size, 1, out_dim))?;

        // Batch matrix multiplication: [B, in_dim, 1] @ [B, 1, out_dim] -> [B, in_dim, out_dim]
        x_b.matmul(&diff_b)
    }

    /// Evaluates directional curvature: Hessian-vector product (H @ v)
    /// For quadratic loss L(x) = 0.5 * ||x @ W||^2, Hessian H = W @ W^T, so H @ v = (v @ W) @ W^T
    pub fn linear_hvp(v: &Tensor, w: &Tensor) -> Result<Tensor> {
        let proj = v.matmul(w)?;
        let w_t = w.t()?;
        proj.matmul(&w_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_forward_jvp_computation() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0f32]], &device)?;
        let v = Tensor::new(&[[0.1f32, 0.2f32]], &device)?;
        let w = Tensor::new(&[[2.0f32], [3.0f32]], &device)?;
        let b = Tensor::new(&[0.5f32], &device)?;

        let (y, jvp) = ForwardAdJvp::linear_jvp(&x, &v, &w, &b)?;
        let y_vec = y.to_vec2::<f32>()?;
        let jvp_vec = jvp.to_vec2::<f32>()?;

        assert_eq!(y_vec, vec![vec![8.5f32]]);
        assert!((jvp_vec[0][0] - 0.8f32).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_per_sample_grad_shape() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0f32], [3.0f32, 4.0f32]], &device)?;
        let pred = Tensor::new(&[[1.5f32], [3.5f32]], &device)?;
        let target = Tensor::new(&[[1.0f32], [3.0f32]], &device)?;

        let per_sample = ForwardAdJvp::per_sample_weights_grad(&x, &pred, &target)?;
        assert_eq!(per_sample.dims(), &[2, 2, 1]);
        Ok(())
    }
}
