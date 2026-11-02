//! Category 01 Topic 03: Discrete vs Continuous Latents & Gradient Estimators.
//!
//! Provides native Candle implementations for:
//! 1. Continuous Pathwise Reparameterization (Gaussian VAE: $z = \mu + \sigma \odot \epsilon$).
//! 2. Analytical Gaussian KL Divergence.
//! 3. Discrete Relaxed Gumbel-Softmax (Concrete distribution).
//! 4. Score Function / REINFORCE Bernoulli gradient estimator.
//!
//! # Mathematical Invariants
//! - Variance Scaling: Continuous pathwise reparameterization yields $\mathcal{O}(1)$ gradient variance
//!   bounded by Lipschitz continuity, while score-function estimators exhibit $\mathcal{O}(K)$ variance
//!   requiring baseline variance reduction.
//! - Information Bottleneck: The analytical KL divergence $D_{\text{KL}}(q(z|x) \| \mathcal{N}(0, I))$
//!   is non-negative and strictly zero iff $\mu = 0$ and $\log \sigma^2 = 0$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c01_paradigms_and_dichotomies::t03_discrete_vs_continuous_latents_and_gradient_estimators::GaussianReparameterization;
//!
//! let device = Device::Cpu;
//! let w_mu = Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap();
//! let b_mu = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
//! let w_lv = Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap();
//! let b_lv = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
//! let w_dec = Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap();
//! let b_dec = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
//!
//! let vae = GaussianReparameterization::new(w_mu, b_mu, w_lv, b_lv, w_dec, b_dec);
//! let x = Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap();
//! let eps = Tensor::zeros((2, 2), candle_core::DType::F32, &device).unwrap();
//! let (z, out, kl) = vae.forward(&x, &eps).unwrap();
//! assert_eq!(z.dims(), &[2, 2]);
//! assert_eq!(out.dims(), &[2, 2]);
//! ```

use candle_core::{Result, Tensor};

/// Continuous Pathwise Reparameterization (Kingma & Welling 2013).
#[derive(Debug, Clone)]
pub struct GaussianReparameterization {
    pub w_mu: Tensor,
    pub b_mu: Tensor,
    pub w_logvar: Tensor,
    pub b_logvar: Tensor,
    pub w_dec: Tensor,
    pub b_dec: Tensor,
}

impl GaussianReparameterization {
    pub fn new(
        w_mu: Tensor,
        b_mu: Tensor,
        w_logvar: Tensor,
        b_logvar: Tensor,
        w_dec: Tensor,
        b_dec: Tensor,
    ) -> Self {
        Self {
            w_mu,
            b_mu,
            w_logvar,
            b_logvar,
            w_dec,
            b_dec,
        }
    }

    /// Evaluates mean $\mu$ and log-variance $\log \sigma^2$.
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mu = x.matmul(&self.w_mu)?.broadcast_add(&self.b_mu)?;
        let logvar = x.matmul(&self.w_logvar)?.broadcast_add(&self.b_logvar)?;
        Ok((mu, logvar))
    }

    /// Continuous pathwise reparameterization: $z = \mu + \exp(0.5 \log \sigma^2) \odot \epsilon$.
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor, eps: &Tensor) -> Result<Tensor> {
        let half_logvar = (logvar * 0.5)?;
        let std = half_logvar.exp()?;
        let noise = std.mul(eps)?;
        mu.add(&noise)
    }

    /// Analytical Gaussian KL divergence: $-0.5 \sum (1 + \log \sigma^2 - \mu^2 - \exp(\log \sigma^2))$.
    pub fn kl_divergence(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        let one = Tensor::ones_like(logvar)?;
        let mu_sq = mu.sqr()?;
        let var = logvar.exp()?;
        let inner = (one.add(logvar)?.sub(&mu_sq)?.sub(&var))?;
        let sum_inner = inner.sum(1)?;
        &sum_inner * (-0.5)
    }

    /// Decodes latent vector: $z W_{\text{dec}} + b_{\text{dec}}$.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let proj = z.matmul(&self.w_dec)?;
        proj.broadcast_add(&self.b_dec)
    }

    /// Full forward evaluation returning $(z, \hat{x}, D_{\text{KL}})$.
    pub fn forward(&self, x: &Tensor, eps: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (mu, logvar) = self.encode(x)?;
        let z = self.reparameterize(&mu, &logvar, eps)?;
        let out = self.decode(&z)?;
        let kl = Self::kl_divergence(&mu, &logvar)?;
        Ok((z, out, kl))
    }
}

/// Discrete Relaxed Gumbel-Softmax (Concrete distribution).
#[derive(Debug, Clone)]
pub struct GumbelSoftmaxRelaxation {
    pub w_logits: Tensor,
    pub b_logits: Tensor,
    pub w_dec: Tensor,
    pub b_dec: Tensor,
    pub temperature: f64,
}

impl GumbelSoftmaxRelaxation {
    pub fn new(
        w_logits: Tensor,
        b_logits: Tensor,
        w_dec: Tensor,
        b_dec: Tensor,
        temperature: f64,
    ) -> Self {
        Self {
            w_logits,
            b_logits,
            w_dec,
            b_dec,
            temperature,
        }
    }

    /// Evaluates relaxed Gumbel-Softmax samples: $\text{softmax}((logits + g) / \tau)$.
    pub fn sample(&self, x: &Tensor, gumbel_noise: &Tensor) -> Result<Tensor> {
        let logits = x.matmul(&self.w_logits)?.broadcast_add(&self.b_logits)?;
        let perturbed = logits.add(gumbel_noise)?;
        let scaled = (&perturbed * (1.0 / self.temperature))?;
        let max_val = scaled.max_keepdim(1)?;
        let centered = scaled.broadcast_sub(&max_val)?;
        let exp_vals = centered.exp()?;
        let sum_exp = exp_vals.sum_keepdim(1)?;
        exp_vals.broadcast_div(&sum_exp)
    }

    /// Decodes relaxed one-hot representation: $y W_{\text{dec}} + b_{\text{dec}}$.
    pub fn decode(&self, y: &Tensor) -> Result<Tensor> {
        let proj = y.matmul(&self.w_dec)?;
        proj.broadcast_add(&self.b_dec)
    }

    /// Full forward evaluation returning $(y_{\text{soft}}, \hat{x})$.
    pub fn forward(&self, x: &Tensor, gumbel_noise: &Tensor) -> Result<(Tensor, Tensor)> {
        let y = self.sample(x, gumbel_noise)?;
        let out = self.decode(&y)?;
        Ok((y, out))
    }
}

/// Score function / REINFORCE gradient estimator.
pub struct BernoulliReinforceEstimator;

impl BernoulliReinforceEstimator {
    /// Computes score function gradient: $(reward - baseline) (b - p)$.
    pub fn score_gradient(
        probs: &Tensor,
        b: &Tensor,
        reward: &Tensor,
        baseline: &Tensor,
    ) -> Result<Tensor> {
        let advantage = reward.broadcast_sub(baseline)?;
        let residual = b.sub(probs)?;
        let adv_2d = advantage.unsqueeze(1)?;
        residual.broadcast_mul(&adv_2d)
    }
}
