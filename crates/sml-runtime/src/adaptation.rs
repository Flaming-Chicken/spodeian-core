//! Online adaptation, Recursive Least Squares (RLS), and Dual-Speed Test-Time Adaptation (TTA).

use serde::{Deserialize, Serialize};

/// Fast Recursive Least Squares (RLS) filter for streaming tick-by-tick adaptation.
///
/// Implements $O(d^2)$ rank-1 Sherman-Morrison covariance updates with an exponential
/// forgetting factor $\lambda \in (0, 1]$.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineRlsHead {
    pub dim: usize,
    pub weights: Vec<f32>,
    pub p_matrix: Vec<f32>,
    pub lambda: f32,
}

impl OnlineRlsHead {
    /// Initialize filter with $d$ dimensions, forgetting factor $\lambda$, and covariance scale $\delta$.
    pub fn new(dim: usize, lambda: f32, delta: f32) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert!(lambda > 0.0 && lambda <= 1.0, "Lambda must be in (0, 1]");
        let mut p = vec![0.0f32; dim * dim];
        for i in 0..dim {
            p[i * dim + i] = delta;
        }

        Self {
            dim,
            weights: vec![0.0f32; dim],
            p_matrix: p,
            lambda,
        }
    }

    /// Predict scalar target: $\hat{y} = \mathbf{w}^T \mathbf{x}$.
    pub fn predict(&self, x: &[f32]) -> f32 {
        assert_eq!(x.len(), self.dim, "Feature dimension mismatch");
        x.iter().zip(&self.weights).map(|(&xi, &wi)| xi * wi).sum()
    }

    /// Step the filter with incoming observation $(\mathbf{x}_t, y_t)$ and return prediction error $e_t = y_t - \hat{y}_t$.
    pub fn update(&mut self, x: &[f32], y: f32) -> f32 {
        let d = self.dim;
        assert_eq!(x.len(), d);

        // 1. Predicted value
        let y_pred = self.predict(x);
        let error = y - y_pred;

        // 2. Compute P * x
        let mut px = vec![0.0f32; d];
        for r in 0..d {
            let row_offset = r * d;
            let mut sum = 0.0f32;
            for c in 0..d {
                sum += self.p_matrix[row_offset + c] * x[c];
            }
            px[r] = sum;
        }

        // 3. Denominator: lambda + x^T * P * x
        let mut xt_px = 0.0f32;
        for c in 0..d {
            xt_px += x[c] * px[c];
        }
        let denom = self.lambda + xt_px;

        // 4. Kalman gain: k = (P * x) / denom
        let mut k = vec![0.0f32; d];
        for r in 0..d {
            k[r] = px[r] / denom;
        }

        // 5. Weight update: w = w + k * error
        for r in 0..d {
            self.weights[r] += k[r] * error;
        }

        // 6. Covariance update: P = (P - k * (x^T * P)) / lambda
        let inv_lambda = 1.0f32 / self.lambda;
        for r in 0..d {
            let row_offset = r * d;
            for c in 0..d {
                self.p_matrix[row_offset + c] =
                    (self.p_matrix[row_offset + c] - k[r] * px[c]) * inv_lambda;
            }
        }

        error
    }
}

/// 64-bit precision Recursive Least Squares (RLS) filter for high-precision quantitative econometrics.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineRlsFilter64 {
    pub num_features: usize,
    pub weights: Vec<f64>,
    pub p_matrix: Vec<f64>,
    pub lambda: f64,
}

#[doc = "**Stability: Experimental** — API subject to change"]
pub type OnlineRlsFilter = OnlineRlsFilter64;

impl OnlineRlsFilter64 {
    /// Initialize filter with $k$ features, forgetting factor $\lambda$, and covariance scale $\delta$.
    pub fn new(num_features: usize, lambda: f64, delta: f64) -> Self {
        assert!(num_features > 0, "Number of features must be positive");
        assert!(lambda > 0.0 && lambda <= 1.0, "Lambda must be in (0, 1]");
        let mut p = vec![0.0f64; num_features * num_features];
        for i in 0..num_features {
            p[i * num_features + i] = delta;
        }

        Self {
            num_features,
            weights: vec![0.0f64; num_features],
            p_matrix: p,
            lambda,
        }
    }

    /// Step the filter with incoming feature vector $\mathbf{x}_t$ and observed return $y_t$.
    pub fn update(&mut self, x: &[f64], y: f64) -> f64 {
        let k = self.num_features;
        assert_eq!(x.len(), k);

        let y_pred: f64 = x.iter().zip(&self.weights).map(|(&xi, &wi)| xi * wi).sum();
        let error = y - y_pred;

        let mut px = vec![0.0f64; k];
        for r in 0..k {
            let row_offset = r * k;
            let mut sum = 0.0f64;
            for c in 0..k {
                sum += self.p_matrix[row_offset + c] * x[c];
            }
            px[r] = sum;
        }

        let mut xt_px = 0.0f64;
        for c in 0..k {
            xt_px += x[c] * px[c];
        }
        let denom = self.lambda + xt_px;

        let mut g = vec![0.0f64; k];
        for r in 0..k {
            g[r] = px[r] / denom;
        }

        for r in 0..k {
            self.weights[r] += g[r] * error;
        }

        let inv_lambda = 1.0f64 / self.lambda;
        for r in 0..k {
            let row_offset = r * k;
            for c in 0..k {
                self.p_matrix[row_offset + c] =
                    (self.p_matrix[row_offset + c] - g[r] * px[c]) * inv_lambda;
            }
        }

        error
    }

    /// Predict scalar target: $\hat{y} = \mathbf{w}^T \mathbf{x}$.
    pub fn predict(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.num_features);
        x.iter().zip(&self.weights).map(|(&xi, &wi)| xi * wi).sum()
    }
}

/// Experience replay memory buffer for cross-regime memory preservation.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceReplayBuffer {
    pub capacity: usize,
    pub buffer: Vec<(Vec<f64>, f64)>,
}

impl ExperienceReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, features: Vec<f64>, actual_return: f64) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push((features, actual_return));
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// 64-bit Sharpness-Aware Minimization (SAM) optimizer for high-precision econometric models.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone)]
pub struct SharpnessAwareMinimizer64 {
    pub rho: f64,
    pub lr: f64,
}

impl SharpnessAwareMinimizer64 {
    pub fn new(rho: f64, lr: f64) -> Self {
        assert!(rho > 0.0, "Rho must be strictly positive");
        assert!(lr > 0.0, "Learning rate must be positive");
        Self { rho, lr }
    }

    /// Compute adversarial weight perturbation vector $\boldsymbol{\epsilon}^*$.
    pub fn compute_perturbation(&self, grad: &[f64]) -> Vec<f64> {
        let norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt().max(1e-8);
        let scale = self.rho / norm;
        grad.iter().map(|&g| g * scale).collect()
    }

    /// Apply SAM weight update given perturbed gradients.
    pub fn update_weights(&self, weights: &mut [f64], perturbed_grad: &[f64]) {
        assert_eq!(weights.len(), perturbed_grad.len());
        for (w, &g) in weights.iter_mut().zip(perturbed_grad.iter()) {
            *w -= self.lr * g;
        }
    }

    /// Clip gradient in-place if its norm exceeds `max_grad_norm`.
    pub fn clip_gradient(&self, grad: &mut [f64], max_grad_norm: f64) -> f64 {
        let norm = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if norm > max_grad_norm && norm > 1e-12 {
            let scale = max_grad_norm / norm;
            for g in grad.iter_mut() {
                *g *= scale;
            }
        }
        norm
    }
}

/// Sharpness-Aware Minimization (SAM) perturbation and gradient clipping utility.
///
/// Computes flat-basin gradient perturbations:
/// $$\boldsymbol{\epsilon}^* = \rho \frac{\nabla_{\mathbf{w}} L}{\|\nabla_{\mathbf{w}} L\|_2 + 10^{-12}}$$
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone)]
pub struct SharpnessAwareMinimizer {
    pub rho: f32,
    pub max_grad_norm: f32,
}

impl SharpnessAwareMinimizer {
    pub fn new(rho: f32, max_grad_norm: f32) -> Self {
        Self { rho, max_grad_norm }
    }

    /// Compute L2 norm of gradient slice.
    pub fn grad_norm(grad: &[f32]) -> f32 {
        let sum_sq: f32 = grad.iter().map(|&g| g * g).sum();
        sum_sq.sqrt()
    }

    /// Clip gradient in-place if its norm exceeds `max_grad_norm`.
    pub fn clip_gradient(&self, grad: &mut [f32]) -> f32 {
        let norm = Self::grad_norm(grad);
        if norm > self.max_grad_norm && norm > 1e-12 {
            let scale = self.max_grad_norm / norm;
            for g in grad.iter_mut() {
                *g *= scale;
            }
        }
        norm
    }

    /// Compute adversarial weight perturbation vector $\boldsymbol{\epsilon}^*$.
    pub fn compute_perturbation(&self, grad: &[f32]) -> Vec<f32> {
        let norm = Self::grad_norm(grad);
        if norm < 1e-12 {
            return vec![0.0f32; grad.len()];
        }
        let scale = self.rho / norm;
        grad.iter().map(|&g| g * scale).collect()
    }
}

/// Continuous Dual-Speed Test-Time Adaptation Manager.
///
/// Harmonizes two timescales of learning:
/// 1. **Slow Encoder Adaptation**: Microscopic learning rate $\eta_{\text{slow}}$,
///    protected by gradient clipping and flat-basin SAM to prevent catastrophic forgetting.
/// 2. **Fast Head Adaptation**: Instantaneous tick-by-tick recursive least squares (RLS)
///    adapting to non-stationary market drift in real time.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone)]
pub struct DualSpeedAdaptationManager {
    pub rls_head: OnlineRlsHead,
    pub sam: SharpnessAwareMinimizer,
    pub slow_lr: f32,
    pub rolling_error_mean: f32,
    pub rolling_error_var: f32,
    pub anomaly_threshold_std: f32,
    pub step_count: usize,
}

impl DualSpeedAdaptationManager {
    pub fn new(dim: usize, slow_lr: f32, fast_lambda: f32, rho: f32, max_grad_norm: f32) -> Self {
        Self {
            rls_head: OnlineRlsHead::new(dim, fast_lambda, 1000.0),
            sam: SharpnessAwareMinimizer::new(rho, max_grad_norm),
            slow_lr,
            rolling_error_mean: 0.0,
            rolling_error_var: 1.0,
            anomaly_threshold_std: 3.0,
            step_count: 0,
        }
    }

    /// Process a streaming tick with input representation $\mathbf{z}_t \in \mathbb{R}^d$ and actual outcome $y_t$.
    ///
    /// Returns `(prediction, error, is_anomaly, training_utility)`.
    pub fn step(&mut self, z: &[f32], y: f32) -> (f32, f32, bool, f32) {
        self.step_count += 1;

        // 1. Fast RLS inference & update
        let pred = self.rls_head.predict(z);
        let error = self.rls_head.update(z, y);

        // 2. Track rolling error distribution (Welford-style / exponential)
        let alpha = 0.05f32;
        let delta = error.abs() - self.rolling_error_mean;
        self.rolling_error_mean += alpha * delta;
        self.rolling_error_var = (1.0 - alpha) * self.rolling_error_var + alpha * delta * delta;
        let std_dev = self.rolling_error_var.sqrt().max(1e-6);

        // 3. Check for regime anomaly
        let z_score = (error.abs() - self.rolling_error_mean) / std_dev;
        let is_anomaly = z_score > self.anomaly_threshold_std;

        // 4. Compute training utility metric U_t in [0, 1]
        let training_utility = (z_score / 5.0).clamp(0.0, 1.0);

        (pred, error, is_anomaly, training_utility)
    }

    /// Perform a slow structural gradient update on the deep representation weights
    /// using clipped gradients and flat-basin SAM regularization.
    pub fn slow_encoder_step(&self, weights: &mut [f32], grad: &mut [f32]) {
        // Apply SAM gradient clipping
        self.sam.clip_gradient(grad);

        // Microscopic parameter update
        for (w, &g) in weights.iter_mut().zip(grad.iter()) {
            *w -= self.slow_lr * g;
        }
    }
}
