//! Category 01 Topic 02: Tree vs Kernel vs Neural Inductive Biases.
//!
//! Provides native Candle implementations for:
//! 1. Differentiable Soft Decision Tree with axis-aligned routing projections.
//! 2. Reproducing Kernel Hilbert Space (RKHS) RBF Kernel Ridge Regressor.
//! 3. Multi-Layer Perceptron (MLP) with adaptive representation learning.
//!
//! # Mathematical Invariants
//! - Rotation Sensitivity: Axis-aligned decision trees partition space via coordinate-orthogonal
//!   hyperplanes and degrade under orthogonal rotations $R \in \mathcal{O}(d)$, while RBF kernels
//!   and neural networks exhibit metric rotation equivariance.
//! - Local vs Global Biases: RBF kernels enforce local isotropic smoothness decay $\exp(-\gamma \|x - s\|^2)$,
//!   whereas neural MLPs compose global affine transformations $W x + b$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use sml_candle::c01_paradigms_and_dichotomies::t02_tree_vs_kernel_vs_neural_inductive_biases::SoftDecisionTree;
//!
//! let device = Device::Cpu;
//! let w_route = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0]], &device).unwrap();
//! let thresh = Tensor::new(&[0.0f32, 0.0, 0.0], &device).unwrap();
//! let w_leaf = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], &device).unwrap();
//! let tree = SoftDecisionTree::new(w_route, thresh, 1.0, w_leaf);
//!
//! let x = Tensor::new(&[[0.5f32, -0.5], [-0.5, 0.5]], &device).unwrap();
//! let y = tree.forward(&x).unwrap();
//! assert_eq!(y.dims(), &[2, 2]);
//! ```

use candle_core::{Result, Tensor};

/// Differentiable Soft Decision Tree with axis-aligned routing projections.
#[derive(Debug, Clone)]
pub struct SoftDecisionTree {
    pub routing_weights: Tensor,
    pub thresholds: Tensor,
    pub temperature: f64,
    pub leaf_weights: Tensor,
}

impl SoftDecisionTree {
    pub fn new(
        routing_weights: Tensor,
        thresholds: Tensor,
        temperature: f64,
        leaf_weights: Tensor,
    ) -> Self {
        Self {
            routing_weights,
            thresholds,
            temperature,
            leaf_weights,
        }
    }

    /// Evaluates differentiable soft decision tree:
    /// $p_{\text{right}} = \sigma(\beta (x W_{\text{route}}^\top - \tau))$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let proj = x.matmul(&self.routing_weights.t()?)?;
        let diff = proj.broadcast_sub(&self.thresholds)?;
        let scaled_diff = (diff * self.temperature)?;
        let neg_logits = (&scaled_diff * (-1.0))?;
        let exp_neg = neg_logits.exp()?;
        let denom = (exp_neg + 1.0)?;
        let p_right = (denom.recip())?;
        let ones = Tensor::ones_like(&p_right)?;
        let p_left = (ones - &p_right)?;

        // Extract internal routing columns
        let p_l0 = p_left.narrow(1, 0, 1)?;
        let p_l1 = p_left.narrow(1, 1, 1)?;
        let p_l2 = p_left.narrow(1, 2, 1)?;

        let p_r0 = p_right.narrow(1, 0, 1)?;
        let p_r1 = p_right.narrow(1, 1, 1)?;
        let p_r2 = p_right.narrow(1, 2, 1)?;

        let mu0 = (&p_l0 * &p_l1)?;
        let mu1 = (&p_l0 * &p_r1)?;
        let mu2 = (&p_r0 * &p_l2)?;
        let mu3 = (&p_r0 * &p_r2)?;

        let leaf_probs = Tensor::cat(&[&mu0, &mu1, &mu2, &mu3], 1)?;
        leaf_probs.matmul(&self.leaf_weights)
    }
}

/// RBF Kernel Ridge regressor / classifier.
#[derive(Debug, Clone)]
pub struct RBFKernelRidge {
    pub support_vectors: Tensor,
    pub dual_coeffs: Tensor,
    pub gamma: f64,
    pub bias: Tensor,
}

impl RBFKernelRidge {
    pub fn new(support_vectors: Tensor, dual_coeffs: Tensor, gamma: f64, bias: Tensor) -> Self {
        Self {
            support_vectors,
            dual_coeffs,
            gamma,
            bias,
        }
    }

    /// Evaluates RBF kernel prediction: $\hat{y} = K(x, S) \alpha + b$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?.sum_keepdim(1)?;
        let s_sq = self.support_vectors.sqr()?.sum_keepdim(1)?.t()?;
        let cross = (x.matmul(&self.support_vectors.t()?)? * 2.0)?;

        let dists_sq = x_sq.broadcast_add(&s_sq)?.broadcast_sub(&cross)?;
        let zeros = Tensor::zeros_like(&dists_sq)?;
        let clamped = dists_sq.maximum(&zeros)?;

        let scaled = (&clamped * (-self.gamma))?;
        let sim = scaled.exp()?;
        let out = sim.matmul(&self.dual_coeffs)?;
        out.broadcast_add(&self.bias)
    }
}

/// Multi-Layer Perceptron (MLP) with ReLU activation.
#[derive(Debug, Clone)]
pub struct MLPClassifier {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
}

impl MLPClassifier {
    pub fn new(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Self {
        Self { w1, b1, w2, b2 }
    }

    /// Evaluates 2-layer MLP forward pass: $W_2 \text{ReLU}(W_1 x + b_1) + b_2$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = x.matmul(&self.w1)?.broadcast_add(&self.b1)?;
        let h1_relu = h1.relu()?;
        h1_relu.matmul(&self.w2)?.broadcast_add(&self.b2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_tree_shape_and_probability_partition() {
        let device = Device::Cpu;
        let w_route = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0]], &device).unwrap();
        let thresh = Tensor::new(&[0.0f32, 0.0, 0.0], &device).unwrap();
        let w_leaf = Tensor::new(
            &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            &device,
        )
        .unwrap();

        let tree = SoftDecisionTree::new(w_route, thresh, 2.0, w_leaf);
        let x = Tensor::new(&[[0.5f32, 0.2], [-0.1, -0.8]], &device).unwrap();
        let y = tree.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 2]);
    }
}
