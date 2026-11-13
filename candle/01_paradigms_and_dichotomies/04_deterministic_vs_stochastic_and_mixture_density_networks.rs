//! Category 01 Topic 04: Deterministic vs Stochastic & Mixture Density Networks.
//!
//! Provides native Candle implementations for:
//! 1. Deterministic Conditional Mean Regressor (MSE Point Estimator).
//! 2. Mixture Density Network (Bishop 1994: multimodal conditional Gaussian mixture).
//! 3. Stable Gaussian Mixture Log-Likelihood via LogSumExp.
//! 4. Multi-Quantile Regressor with Pinball / Check Loss.
//!
//! # Mathematical Invariants
//! - Conditional Mean Limitation: Deterministic MSE converges to $\mathbb{E}[Y|X=x]$,
//!   which collapses into low-probability regions on multimodal or bifurcated targets.
//! - Simplex Invariant: Mixture coefficients satisfy $\sum_{k=1}^K \pi_k(x) = 1$ and $\pi_k(x) > 0$.
//! - Positivity Invariant: Gaussian scales satisfy $\sigma_k(x) = \exp(s_k(x)) > 0$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use sml_candle::c01_paradigms_and_dichotomies::t04_deterministic_vs_stochastic_and_mixture_density_networks::DeterministicRegressor;
//!
//! let device = Device::Cpu;
//! let w_h = Tensor::zeros((4, 8), candle_core::DType::F32, &device).unwrap();
//! let b_h = Tensor::zeros(8, candle_core::DType::F32, &device).unwrap();
//! let w_det = Tensor::zeros((8, 1), candle_core::DType::F32, &device).unwrap();
//! let b_det = Tensor::zeros(1, candle_core::DType::F32, &device).unwrap();
//!
//! let model = DeterministicRegressor::new(w_h, b_h, w_det, b_det);
//! let x = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
//! let y = model.forward(&x).unwrap();
//! assert_eq!(y.dims(), &[2, 1]);
//! ```

use candle_core::{Result, Tensor};

/// Deterministic Conditional Mean Regressor (MSE Point Estimator).
#[derive(Debug, Clone)]
pub struct DeterministicRegressor {
    pub w_h: Tensor,
    pub b_h: Tensor,
    pub w_det: Tensor,
    pub b_det: Tensor,
}

impl DeterministicRegressor {
    pub fn new(w_h: Tensor, b_h: Tensor, w_det: Tensor, b_det: Tensor) -> Self {
        Self {
            w_h,
            b_h,
            w_det,
            b_det,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.matmul(&self.w_h)?.broadcast_add(&self.b_h)?.relu()?;
        hidden.matmul(&self.w_det)?.broadcast_add(&self.b_det)
    }
}

/// Mixture Density Network (Bishop 1994).
#[derive(Debug, Clone)]
pub struct MixtureDensityNetwork {
    pub w_h: Tensor,
    pub b_h: Tensor,
    pub w_pi: Tensor,
    pub b_pi: Tensor,
    pub w_mu: Tensor,
    pub b_mu: Tensor,
    pub w_sigma: Tensor,
    pub b_sigma: Tensor,
}

impl MixtureDensityNetwork {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w_h: Tensor,
        b_h: Tensor,
        w_pi: Tensor,
        b_pi: Tensor,
        w_mu: Tensor,
        b_mu: Tensor,
        w_sigma: Tensor,
        b_sigma: Tensor,
    ) -> Self {
        Self {
            w_h,
            b_h,
            w_pi,
            b_pi,
            w_mu,
            b_mu,
            w_sigma,
            b_sigma,
        }
    }

    /// Forward pass extracting mixture parameters (pi, mu, sigma).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let hidden = x.matmul(&self.w_h)?.broadcast_add(&self.b_h)?.relu()?;
        let pi_logits = hidden.matmul(&self.w_pi)?.broadcast_add(&self.b_pi)?;
        let pi = candle_nn::ops::softmax(&pi_logits, 1)?;
        let mu = hidden.matmul(&self.w_mu)?.broadcast_add(&self.b_mu)?;
        let log_sigma = hidden.matmul(&self.w_sigma)?.broadcast_add(&self.b_sigma)?;
        let sigma = log_sigma.exp()?;
        Ok((pi, mu, sigma))
    }

    /// Log-likelihood log p(y|x) under Gaussian Mixture.
    pub fn log_prob(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let hidden = x.matmul(&self.w_h)?.broadcast_add(&self.b_h)?.relu()?;
        let pi_logits = hidden.matmul(&self.w_pi)?.broadcast_add(&self.b_pi)?;
        let log_pi = candle_nn::ops::log_softmax(&pi_logits, 1)?;

        let mu = hidden.matmul(&self.w_mu)?.broadcast_add(&self.b_mu)?;
        let log_sigma = hidden.matmul(&self.w_sigma)?.broadcast_add(&self.b_sigma)?;
        let sigma = log_sigma.exp()?;

        let diff = y.broadcast_sub(&mu)?;
        let z = diff.broadcast_div(&sigma)?;
        let z_sq = (z.sqr()? * 0.5)?;

        let half_log_2pi = (2.0f64 * std::f64::consts::PI).ln() * 0.5;
        let log_gauss = log_sigma
            .affine(-1.0, -half_log_2pi)?
            .broadcast_sub(&z_sq)?;

        let terms = log_pi.broadcast_add(&log_gauss)?;
        let max_terms = terms.max_keepdim(1)?;
        let exp_terms = terms.broadcast_sub(&max_terms)?.exp()?;
        let sum_exp = exp_terms.sum_keepdim(1)?;
        let lse = sum_exp.log()?.broadcast_add(&max_terms)?;
        lse.squeeze(1)
    }
}

/// Direct Multi-Quantile Regressor (Koenker & Bassett 1978).
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    pub w_h: Tensor,
    pub b_h: Tensor,
    pub w_quantile: Tensor,
    pub b_quantile: Tensor,
}

impl QuantileRegressor {
    pub fn new(w_h: Tensor, b_h: Tensor, w_quantile: Tensor, b_quantile: Tensor) -> Self {
        Self {
            w_h,
            b_h,
            w_quantile,
            b_quantile,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.matmul(&self.w_h)?.broadcast_add(&self.b_h)?.relu()?;
        hidden
            .matmul(&self.w_quantile)?
            .broadcast_add(&self.b_quantile)
    }

    pub fn pinball_loss(y_pred: &Tensor, y_true: &Tensor, quantiles: &Tensor) -> Result<Tensor> {
        let u = y_true.broadcast_sub(y_pred)?;
        let q_u = u.broadcast_mul(quantiles)?;
        let q_minus_1 = quantiles.affine(1.0, -1.0)?;
        let qm1_u = u.broadcast_mul(&q_minus_1)?;
        q_u.maximum(&qm1_u)
    }
}
