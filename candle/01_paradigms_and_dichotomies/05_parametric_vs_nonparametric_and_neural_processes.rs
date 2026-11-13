//! Category 01 Topic 05: Parametric vs Non-Parametric & Neural Processes.
//!
//! Provides native Candle implementations for:
//! 1. Parametric Model: Fixed-capacity 2-layer MLP with $\mathcal{O}(1)$ test memory.
//! 2. Exact Non-Parametric Model: Gaussian Process / Kernel Ridge regression.
//! 3. Semi-Parametric Model: Attentive Neural Process (Kim et al. 2019) with $\mathcal{O}(C \cdot T)$ cross-attention.
//! 4. Semi-Parametric Model: Conditional Neural Process (Garnelo et al. 2018) with uniform mean-pooling.
//!
//! # Mathematical Invariants
//! - Capacity Scaling: Parametric models have fixed $\mathcal{O}(1)$ parameter capacity independent of $N$.
//!   Non-parametric models grow with $N$, incurring $\mathcal{O}(N^3)$ compute and $\mathcal{O}(N^2)$ memory.
//!   Attentive Neural Processes achieve dynamic context adaptation in $\mathcal{O}(C \cdot T)$ compute with $\mathcal{O}(1)$ parameters.
//! - Attention Invariant: The cross-attention weights satisfy $\sum_{c=1}^C A_{t, c} = 1$ and $A_{t, c} > 0$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use sml_candle::c01_paradigms_and_dichotomies::t05_parametric_vs_nonparametric_and_neural_processes::ParametricMLP;
//!
//! let device = Device::Cpu;
//! let w_mlp1 = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
//! let b_mlp1 = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();
//! let w_mlp2 = Tensor::zeros((4, 1), candle_core::DType::F32, &device).unwrap();
//! let b_mlp2 = Tensor::zeros(1, candle_core::DType::F32, &device).unwrap();
//!
//! let model = ParametricMLP::new(w_mlp1, b_mlp1, w_mlp2, b_mlp2);
//! let xt = Tensor::zeros((2, 3, 2), candle_core::DType::F32, &device).unwrap();
//! let y = model.forward(&xt).unwrap();
//! assert_eq!(y.dims(), &[2, 3, 1]);
//! ```

use candle_core::{Result, Tensor};

#[inline]
fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let b = lhs.dim(0)?;
    let rhs_b = rhs.broadcast_as((b, rhs.dim(0)?, rhs.dim(1)?))?;
    lhs.matmul(&rhs_b)
}

/// Parametric Multi-Layer Perceptron (Fixed Memory Capacity).
#[derive(Debug, Clone)]
pub struct ParametricMLP {
    pub w_mlp1: Tensor,
    pub b_mlp1: Tensor,
    pub w_mlp2: Tensor,
    pub b_mlp2: Tensor,
}

impl ParametricMLP {
    pub fn new(w_mlp1: Tensor, b_mlp1: Tensor, w_mlp2: Tensor, b_mlp2: Tensor) -> Self {
        Self {
            w_mlp1,
            b_mlp1,
            w_mlp2,
            b_mlp2,
        }
    }

    pub fn forward(&self, x_target: &Tensor) -> Result<Tensor> {
        let h = matmul_3d_2d(x_target, &self.w_mlp1)?
            .broadcast_add(&self.b_mlp1)?
            .relu()?;
        matmul_3d_2d(&h, &self.w_mlp2)?.broadcast_add(&self.b_mlp2)
    }
}

/// Exact Gaussian Process / Kernel Ridge Regressor (Rasmussen & Williams 2006).
#[derive(Debug, Clone)]
pub struct ExactGaussianProcess {
    pub alpha: Tensor,
    pub lengthscale: f64,
}

impl ExactGaussianProcess {
    pub fn new(alpha: Tensor, lengthscale: f64) -> Self {
        Self { alpha, lengthscale }
    }

    /// Evaluates posterior mean at target coordinates given precomputed dual weights alpha:
    /// $\mu(x_t) = K(x_t, X_c) \alpha$.
    pub fn forward(&self, x_target: &Tensor, x_context: &Tensor) -> Result<Tensor> {
        let xt_norm = x_target.sqr()?.sum_keepdim(2)?; // [B, Nt, 1]
        let xc_norm = x_context.sqr()?.sum_keepdim(2)?; // [B, Nc, 1]
        let cross = x_target.matmul(&x_context.transpose(1, 2)?)?; // [B, Nt, Nc]

        let dist_sq = xt_norm
            .broadcast_add(&xc_norm.transpose(1, 2)?)?
            .broadcast_sub(&(cross * 2.0)?)?
            .clamp(0.0f32, f32::MAX)?;

        let gamma = 0.5 / (self.lengthscale * self.lengthscale);
        let k_tc = (dist_sq * (-gamma))?.exp()?;
        k_tc.matmul(&self.alpha)
    }
}

/// Semi-Parametric Attentive Neural Process (Kim et al. 2019).
#[derive(Debug, Clone)]
pub struct AttentiveNeuralProcess {
    pub w_enc: Tensor,
    pub b_enc: Tensor,
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_dec: Tensor,
    pub b_dec: Tensor,
}

impl AttentiveNeuralProcess {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w_enc: Tensor,
        b_enc: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        w_dec: Tensor,
        b_dec: Tensor,
    ) -> Self {
        Self {
            w_enc,
            b_enc,
            w_q,
            w_k,
            w_v,
            w_dec,
            b_dec,
        }
    }

    pub fn forward(
        &self,
        x_context: &Tensor,
        y_context: &Tensor,
        x_target: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let xy_c = Tensor::cat(&[x_context, y_context], 2)?;
        let v_c = matmul_3d_2d(&xy_c, &self.w_enc)?
            .broadcast_add(&self.b_enc)?
            .relu()?;

        let q = matmul_3d_2d(x_target, &self.w_q)?;
        let k = matmul_3d_2d(x_context, &self.w_k)?;
        let v = matmul_3d_2d(&v_c, &self.w_v)?;

        let dim_h = self.w_q.dim(1)? as f64;
        let scale = 1.0 / dim_h.sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;

        let r_anp = attn_weights.matmul(&v)?;
        let xt_r = Tensor::cat(&[x_target, &r_anp], 2)?;
        let y_pred = matmul_3d_2d(&xt_r, &self.w_dec)?.broadcast_add(&self.b_dec)?;
        Ok((y_pred, attn_weights))
    }
}

/// Semi-Parametric Conditional Neural Process (Garnelo et al. 2018).
#[derive(Debug, Clone)]
pub struct ConditionalNeuralProcess {
    pub w_enc: Tensor,
    pub b_enc: Tensor,
    pub w_dec: Tensor,
    pub b_dec: Tensor,
}

impl ConditionalNeuralProcess {
    pub fn new(w_enc: Tensor, b_enc: Tensor, w_dec: Tensor, b_dec: Tensor) -> Self {
        Self {
            w_enc,
            b_enc,
            w_dec,
            b_dec,
        }
    }

    pub fn forward(
        &self,
        x_context: &Tensor,
        y_context: &Tensor,
        x_target: &Tensor,
    ) -> Result<Tensor> {
        let xy_c = Tensor::cat(&[x_context, y_context], 2)?;
        let v_c = matmul_3d_2d(&xy_c, &self.w_enc)?
            .broadcast_add(&self.b_enc)?
            .relu()?;

        let r_pooled = v_c.mean_keepdim(1)?;
        let r_expanded =
            r_pooled.broadcast_as((x_target.dim(0)?, x_target.dim(1)?, r_pooled.dim(2)?))?;

        let xt_r = Tensor::cat(&[x_target, &r_expanded], 2)?;
        matmul_3d_2d(&xt_r, &self.w_dec)?.broadcast_add(&self.b_dec)
    }
}
