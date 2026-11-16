//! # Spodeian Forecast
//!
//! Pure-Rust time-series forecasting, volatility, and probabilistic calibration primitives:
//! - `holt_winters_additive`: damped/undamped Holt's linear trend exponential smoothing
//! - `KalmanFilter1D`: scalar local-level state-space filter & forecast
//! - `Garch11`: GARCH(1,1) conditional variance recursion & multi-step volatility forecast
//! - `conformal_prediction_interval`, `pinball_loss`, `crps_gaussian`

use serde::{Deserialize, Serialize};
use spodeian_stats::normal_cdf;
use thiserror::Error;

/// Errors raised by forecasting routines.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForecastError {
    #[error("insufficient observations: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Holt's linear trend exponential smoothing (`alpha` level, `beta` trend) returning `horizon` forecasts.
pub fn holt_linear_forecast(
    series: &[f64],
    alpha: f64,
    beta: f64,
    horizon: usize,
) -> Result<Vec<f64>, ForecastError> {
    if series.len() < 3 {
        return Err(ForecastError::InsufficientData {
            required: 3,
            actual: series.len(),
        });
    }
    if !(0.0..=1.0).contains(&alpha) || !(0.0..=1.0).contains(&beta) {
        return Err(ForecastError::InvalidParameter(
            "alpha and beta must be in [0, 1]".to_string(),
        ));
    }
    let mut level = series[0];
    let mut trend = series[1] - series[0];
    for &y in series.iter().skip(1) {
        let prev_level = level;
        level = alpha * y + (1.0 - alpha) * (level + trend);
        trend = beta * (level - prev_level) + (1.0 - beta) * trend;
    }
    Ok((1..=horizon).map(|h| level + (h as f64) * trend).collect())
}

/// Scalar local-level Kalman filter (`x_{t+1} = x_t + w_t`, `y_t = x_t + v_t`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KalmanFilter1D {
    pub state: f64,
    pub variance: f64,
    pub process_noise_q: f64,
    pub measurement_noise_r: f64,
}

impl KalmanFilter1D {
    pub fn new(
        init_state: f64,
        init_var: f64,
        process_noise_q: f64,
        measurement_noise_r: f64,
    ) -> Self {
        Self {
            state: init_state,
            variance: init_var.max(1e-9),
            process_noise_q: process_noise_q.max(0.0),
            measurement_noise_r: measurement_noise_r.max(1e-9),
        }
    }

    /// Runs one predict + update step and returns the filtered `(state, variance)`.
    pub fn update(&mut self, observation: f64) -> (f64, f64) {
        let pred_var = self.variance + self.process_noise_q;
        let gain = pred_var / (pred_var + self.measurement_noise_r);
        self.state += gain * (observation - self.state);
        self.variance = (1.0 - gain) * pred_var;
        (self.state, self.variance)
    }
}

/// GARCH(1,1) conditional variance model: $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Garch11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Garch11 {
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Result<Self, ForecastError> {
        if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0 {
            return Err(ForecastError::InvalidParameter(
                "GARCH(1,1) requires omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1"
                    .to_string(),
            ));
        }
        Ok(Self { omega, alpha, beta })
    }

    /// Conditional variances of `residuals` and `horizon` variance forecasts beyond them.
    ///
    /// Uses $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$, so:
    /// - `filtered[t]` is the variance of `residuals[t]`, which depends only on earlier
    ///   residuals. `filtered[0]` is the sample variance of `residuals`, the usual start-up value.
    /// - `forecasts[0]` is the variance of the first residual after the sample (one step
    ///   ahead), and `forecasts[k]` reverts toward $\omega / (1 - \alpha - \beta)$ at rate
    ///   $\alpha + \beta$.
    pub fn filter_and_forecast(
        &self,
        residuals: &[f64],
        horizon: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), ForecastError> {
        if residuals.len() < 4 {
            return Err(ForecastError::InsufficientData {
                required: 4,
                actual: residuals.len(),
            });
        }
        let long_run_var = self.omega / (1.0 - self.alpha - self.beta);
        let mut h_t = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        if h_t <= 1e-12 {
            h_t = long_run_var;
        }
        let mut series = Vec::with_capacity(residuals.len());
        for &eps in residuals {
            // The variance of this residual is known before it is observed;
            // it then feeds the variance of the next one.
            series.push(h_t);
            h_t = self.omega + self.alpha * eps * eps + self.beta * h_t;
        }
        // `h_t` is now the one-step-ahead variance.
        let mut forecasts = Vec::with_capacity(horizon);
        let mut next_h = h_t;
        for _ in 0..horizon {
            forecasts.push(next_h);
            next_h = self.omega + (self.alpha + self.beta) * next_h;
        }
        Ok((series, forecasts))
    }
}

/// Split-conformal prediction interval `(lower, upper)` around `point_forecast` at miscoverage `alpha`.
pub fn conformal_prediction_interval(
    point_forecast: f64,
    calibration_abs_residuals: &[f64],
    alpha: f64,
) -> Result<(f64, f64), ForecastError> {
    if calibration_abs_residuals.is_empty() {
        return Err(ForecastError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if !(0.0..1.0).contains(&alpha) || alpha == 0.0 {
        return Err(ForecastError::InvalidParameter(
            "alpha must be strictly in (0, 1)".to_string(),
        ));
    }
    let mut sorted: Vec<f64> = calibration_abs_residuals.iter().map(|r| r.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let rank = (((1.0 - alpha) * (n as f64 + 1.0)).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let q = sorted[rank];
    Ok((point_forecast - q, point_forecast + q))
}

/// Pinball (quantile) loss at quantile level `tau` in `(0, 1)`.
pub fn pinball_loss(y_true: f64, y_pred: f64, tau: f64) -> f64 {
    let diff = y_true - y_pred;
    if diff >= 0.0 {
        tau * diff
    } else {
        (tau - 1.0) * diff
    }
}

/// Closed-form Continuous Ranked Probability Score (CRPS) for a Gaussian predictive distribution $\mathcal{N}(\mu, \sigma^2)$.
pub fn crps_gaussian(y_true: f64, mu: f64, sigma: f64) -> f64 {
    let s = sigma.max(1e-9);
    let z = (y_true - mu) / s;
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = normal_cdf(z);
    s * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / std::f64::consts::PI.sqrt())
}
