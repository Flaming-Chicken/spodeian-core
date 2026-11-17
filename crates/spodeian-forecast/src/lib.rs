//! # Spodeian Forecast
//!
//! Pure-Rust time-series forecasting, volatility, and probabilistic calibration primitives:
//! - `holt_winters_additive`: damped/undamped Holt's linear trend exponential smoothing
//! - `KalmanFilter1D`: scalar local-level state-space filter & forecast
//! - `Garch11`: GARCH(1,1) conditional variance recursion & multi-step volatility forecast
//! - `Garch11::fit` (quasi-maximum likelihood) and `fit_holt_linear` (least squares): parameter estimation
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

/// Result of fitting a [`Garch11`] by quasi-maximum likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GarchFit {
    pub model: Garch11,
    /// Gaussian log-likelihood at the fitted parameters.
    pub log_likelihood: f64,
    /// Optimiser iterations used by the best of the starting points.
    pub iterations: usize,
}

impl Garch11 {
    /// Gaussian log-likelihood of `residuals` under this model.
    ///
    /// The recursion starts from the sample variance, the same start-up value as
    /// [`Garch11::filter_and_forecast`].
    pub fn log_likelihood(&self, residuals: &[f64]) -> Result<f64, ForecastError> {
        if residuals.len() < 4 {
            return Err(ForecastError::InsufficientData {
                required: 4,
                actual: residuals.len(),
            });
        }
        let start = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        if start <= 1e-300 {
            return Err(ForecastError::InvalidParameter(
                "residuals have zero variance".to_string(),
            ));
        }
        let ln_two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut h = start;
        let mut log_likelihood = 0.0;
        for &e in residuals {
            let variance = h.max(1e-300);
            log_likelihood += -0.5 * (ln_two_pi + variance.ln() + e * e / variance);
            h = self.omega + self.alpha * e * e + self.beta * h;
        }
        Ok(log_likelihood)
    }

    /// Fit by Gaussian quasi-maximum likelihood with variance targeting.
    ///
    /// `omega` is set so the unconditional variance equals the sample variance, which leaves
    /// `alpha` and `beta` to search. They are mapped through a logistic transform so every
    /// candidate is a valid stationary model, and optimised with Nelder-Mead from several
    /// starting points (the likelihood is flat along the persistence ridge, so one start can
    /// stop early). Needs at least 50 observations; expect wide uncertainty below a few hundred.
    pub fn fit(residuals: &[f64]) -> Result<GarchFit, ForecastError> {
        const MIN_OBSERVATIONS: usize = 50;
        if residuals.len() < MIN_OBSERVATIONS {
            return Err(ForecastError::InsufficientData {
                required: MIN_OBSERVATIONS,
                actual: residuals.len(),
            });
        }
        let target = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        if target <= 1e-300 {
            return Err(ForecastError::InvalidParameter(
                "residuals have zero variance".to_string(),
            ));
        }
        const MAX_PERSISTENCE: f64 = 0.9999;
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let logit = |p: f64| (p / (1.0 - p)).ln();
        let model_at = |u: [f64; 2]| -> Option<Garch11> {
            let persistence = MAX_PERSISTENCE * sigmoid(u[0]);
            let share = sigmoid(u[1]);
            Garch11::new(
                target * (1.0 - persistence),
                persistence * share,
                persistence * (1.0 - share),
            )
            .ok()
        };
        let objective = |u: [f64; 2]| -> f64 {
            model_at(u)
                .and_then(|m| m.log_likelihood(residuals).ok())
                .map_or(f64::INFINITY, |ll| -ll)
        };

        let starts = [(0.95, 0.10), (0.90, 0.20), (0.98, 0.05), (0.70, 0.30)];
        let mut best: Option<([f64; 2], f64, usize)> = None;
        for (persistence, share) in starts {
            let start = [logit(persistence / MAX_PERSISTENCE), logit(share)];
            let (point, value, iterations) = nelder_mead_2d(&objective, start, 0.5, 600, 1e-12);
            if best.is_none_or(|(_, v, _)| value < v) {
                best = Some((point, value, iterations));
            }
        }
        let (point, value, iterations) = best.ok_or_else(|| {
            ForecastError::InvalidParameter("optimiser produced no result".to_string())
        })?;
        let model = model_at(point).ok_or_else(|| {
            ForecastError::InvalidParameter("optimiser left the parameter space".to_string())
        })?;
        Ok(GarchFit {
            model,
            log_likelihood: -value,
            iterations,
        })
    }
}

/// Minimise `f` over two parameters with the Nelder-Mead simplex method.
/// Returns the best point, its value and the iterations used.
fn nelder_mead_2d(
    f: &dyn Fn([f64; 2]) -> f64,
    start: [f64; 2],
    step: f64,
    max_iterations: usize,
    tolerance: f64,
) -> ([f64; 2], f64, usize) {
    let mut simplex = [
        start,
        [start[0] + step, start[1]],
        [start[0], start[1] + step],
    ];
    let mut values = [f(simplex[0]), f(simplex[1]), f(simplex[2])];
    let mut iterations = 0;
    while iterations < max_iterations {
        iterations += 1;
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        simplex = order.map(|i| simplex[i]);
        values = order.map(|i| values[i]);
        if (values[2] - values[0]).abs() <= tolerance * (1.0 + values[0].abs()) {
            break;
        }
        let centroid = [
            (simplex[0][0] + simplex[1][0]) / 2.0,
            (simplex[0][1] + simplex[1][1]) / 2.0,
        ];
        // centroid + t * (worst - centroid): t = -1 reflects, -2 expands, -0.5 / 0.5 contract.
        let along = |t: f64| {
            [
                centroid[0] + t * (simplex[2][0] - centroid[0]),
                centroid[1] + t * (simplex[2][1] - centroid[1]),
            ]
        };
        let reflected = along(-1.0);
        let f_reflected = f(reflected);
        if f_reflected < values[0] {
            let expanded = along(-2.0);
            let f_expanded = f(expanded);
            if f_expanded < f_reflected {
                simplex[2] = expanded;
                values[2] = f_expanded;
            } else {
                simplex[2] = reflected;
                values[2] = f_reflected;
            }
        } else if f_reflected < values[1] {
            simplex[2] = reflected;
            values[2] = f_reflected;
        } else {
            let contracted = if f_reflected < values[2] {
                along(-0.5)
            } else {
                along(0.5)
            };
            let f_contracted = f(contracted);
            if f_contracted < values[2].min(f_reflected) {
                simplex[2] = contracted;
                values[2] = f_contracted;
            } else {
                for i in 1..3 {
                    simplex[i] = [
                        simplex[0][0] + 0.5 * (simplex[i][0] - simplex[0][0]),
                        simplex[0][1] + 0.5 * (simplex[i][1] - simplex[0][1]),
                    ];
                    values[i] = f(simplex[i]);
                }
            }
        }
    }
    let best = (0..3)
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .unwrap_or(0);
    (simplex[best], values[best], iterations)
}

/// Result of fitting Holt's linear trend smoothing by least squares.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HoltFit {
    pub alpha: f64,
    pub beta: f64,
    /// Sum of squared one-step-ahead forecast errors at the fitted parameters.
    pub sse: f64,
}

/// Sum of squared one-step-ahead errors of Holt's linear trend method, started as in
/// [`holt_linear_forecast`] (level `series[0]`, trend `series[1] - series[0]`).
pub fn holt_one_step_sse(series: &[f64], alpha: f64, beta: f64) -> f64 {
    let mut level = series[0];
    let mut trend = series[1] - series[0];
    let mut sse = 0.0;
    for &y in series.iter().skip(1) {
        let error = y - (level + trend);
        sse += error * error;
        let previous_level = level;
        level = alpha * y + (1.0 - alpha) * (level + trend);
        trend = beta * (level - previous_level) + (1.0 - beta) * trend;
    }
    sse
}

/// Fit the smoothing parameters of Holt's linear trend method by minimising the squared
/// one-step-ahead errors: a 21 x 21 grid over `[0, 1]^2`, refined by a shrinking compass
/// search. Needs at least 5 finite observations.
pub fn fit_holt_linear(series: &[f64]) -> Result<HoltFit, ForecastError> {
    if series.len() < 5 {
        return Err(ForecastError::InsufficientData {
            required: 5,
            actual: series.len(),
        });
    }
    if series.iter().any(|v| !v.is_finite()) {
        return Err(ForecastError::InvalidParameter(
            "series must be finite".to_string(),
        ));
    }
    let mut best = (0.0, 0.0, f64::INFINITY);
    for i in 0..=20 {
        for j in 0..=20 {
            let (alpha, beta) = (i as f64 / 20.0, j as f64 / 20.0);
            let sse = holt_one_step_sse(series, alpha, beta);
            if sse < best.2 {
                best = (alpha, beta, sse);
            }
        }
    }
    let (mut alpha, mut beta, mut sse) = best;
    let mut step = 0.025;
    while step > 1e-9 {
        let mut improved = false;
        for (da, db) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
            let (a, b) = ((alpha + da).clamp(0.0, 1.0), (beta + db).clamp(0.0, 1.0));
            let candidate = holt_one_step_sse(series, a, b);
            if candidate < sse {
                (alpha, beta, sse) = (a, b, candidate);
                improved = true;
            }
        }
        if !improved {
            step /= 2.0;
        }
    }
    Ok(HoltFit { alpha, beta, sse })
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
