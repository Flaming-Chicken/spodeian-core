//! Category 01 Topic 01: Generative vs Discriminative Classifiers.
//!
//! Provides native Candle implementations for:
//! 1. Generative Gaussian Discriminant Analysis (GDA) linear logit evaluation:
//!    $$\mathbf{\eta}_c = \mathbf{x}^\top \mathbf{w}_c + b_c$$
//!    where $\mathbf{w}_c = \mathbf{\Sigma}^{-1} \mathbf{\mu}_c$ and $b_c = -\frac{1}{2}\mathbf{\mu}_c^\top \mathbf{\Sigma}^{-1} \mathbf{\mu}_c + \ln \pi_c$.
//! 2. Discriminative Linear Softmax Classifier.
//! 3. Posterior probability evaluation via numerically stable Softmax.
//!
//! # Mathematical Invariants
//! - Duality: When class-conditional distributions are normal with shared covariance $\mathcal{N}(\mathbf{\mu}_c, \mathbf{\Sigma})$,
//!   generative GDA induces the exact same linear functional form as discriminative logistic regression.
//! - Sample Complexity: Generative models reach asymptotic efficiency in $\mathcal{O}(\log D)$ samples,
//!   while discriminative models require $\mathcal{O}(D)$ samples but achieve lower asymptotic error under misspecification.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c01_paradigms_and_dichotomies::t01_generative_vs_discriminative_classifiers::{
//!     DiscriminativeClassifier, GaussianDiscriminantAnalysis,
//! };
//!
//! let device = Device::Cpu;
//! let w_derived = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
//! let b_derived = Tensor::new(&[0.0f32, 0.0], &device).unwrap();
//! let gda = GaussianDiscriminantAnalysis::new(w_derived, b_derived);
//!
//! let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
//! let logits = gda.predict_logits(&x).unwrap();
//! assert_eq!(logits.dims(), &[2, 2]);
//! ```

use candle_core::{Device, Result, Tensor};

/// Generative Gaussian Discriminant Analysis (GDA) predictor.
#[derive(Debug, Clone)]
pub struct GaussianDiscriminantAnalysis {
    pub w_derived: Tensor,
    pub b_derived: Tensor,
}

impl GaussianDiscriminantAnalysis {
    pub fn new(w_derived: Tensor, b_derived: Tensor) -> Self {
        Self {
            w_derived,
            b_derived,
        }
    }

    /// Evaluates linear discriminant logits: $x @ w_{\text{derived}} + b_{\text{derived}}$.
    pub fn predict_logits(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.matmul(&self.w_derived)?;
        logits.broadcast_add(&self.b_derived)
    }

    /// Evaluates posterior class probabilities via numerically stable Softmax.
    pub fn predict_proba(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let logits = self.predict_logits(x)?;
        stable_softmax(&logits, device)
    }
}

/// Discriminative Linear Softmax Classifier.
#[derive(Debug, Clone)]
pub struct DiscriminativeClassifier {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl DiscriminativeClassifier {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Forward affine projection: $x @ weight + bias$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight)?;
        y.broadcast_add(&self.bias)
    }

    /// Evaluates posterior class probabilities via numerically stable Softmax.
    pub fn predict_proba(&self, x: &Tensor, device: &Device) -> Result<Tensor> {
        let logits = self.forward(x)?;
        stable_softmax(&logits, device)
    }
}

/// Numerically stable Softmax along the last dimension of a 2D tensor.
pub fn stable_softmax(x: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = x.dims();
    let batch = dims[0];
    let num_classes = dims[1];
    let x_flat = x.to_vec2::<f32>()?;

    let mut probs = vec![0.0f32; batch * num_classes];
    for i in 0..batch {
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            if x_flat[i][c] > max_val {
                max_val = x_flat[i][c];
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let exp_val = (x_flat[i][c] - max_val).exp();
            probs[i * num_classes + c] = exp_val;
            sum_exp += exp_val;
        }

        let inv_sum = 1.0f32 / (sum_exp + 1e-12);
        for c in 0..num_classes {
            probs[i * num_classes + c] *= inv_sum;
        }
    }

    Tensor::from_vec(probs, (batch, num_classes), device)
}
