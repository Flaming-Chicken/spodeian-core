//! # Spodeian Stats
//!
//! Pure-Rust statistical distributions, quantiles, linear/robust regressions,
//! HAC standard errors, stationarity tests, and cointegration diagnostics.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors returned by statistical routines.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StatsError {
    #[error("insufficient observations: need at least {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("singular or ill-conditioned matrix")]
    SingularMatrix,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Regression output with OLS/WLS/Ridge/Huber coefficients and optional HAC standard errors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionResult {
    pub beta: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub r_squared: f64,
    pub residuals: Vec<f64>,
}

/// Standard normal cumulative distribution function $\Phi(x)$ via Abramowitz-Stegun 7.1.26.
pub fn normal_cdf(x: f64) -> f64 {
    let z = x / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if z >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

/// Inverse standard normal CDF $\Phi^{-1}(p)$ using Acklam's rational approximation.
pub fn normal_ppf(p: f64) -> Result<f64, StatsError> {
    if !(0.0..1.0).contains(&p) || p == 0.0 {
        return Err(StatsError::InvalidParameter(
            "probability p must be strictly in (0, 1)".to_string(),
        ));
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        let num = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q) + C[5];
        let den = ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q) + 1.0;
        Ok(num / den)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        let num = ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r) + A[5]) * q;
        let den = (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r) + 1.0;
        Ok(num / den)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q) + C[5];
        let den = ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q) + 1.0;
        Ok(-num / den)
    }
}

fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9)
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * z).sin().abs().ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Student's t cumulative distribution function for `df > 0` via adaptive Simpson quadrature of the symmetric t-PDF.
pub fn student_t_cdf(t: f64, df: f64) -> Result<f64, StatsError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(StatsError::InvalidParameter(
            "degrees of freedom must be positive and finite".to_string(),
        ));
    }
    if t == 0.0 {
        return Ok(0.5);
    }
    if df > 1_000.0 {
        return Ok(normal_cdf(t));
    }
    let log_norm =
        ln_gamma(0.5 * (df + 1.0)) - ln_gamma(0.5 * df) - 0.5 * (df * std::f64::consts::PI).ln();
    let pdf = |u: f64| -> f64 { (log_norm - 0.5 * (df + 1.0) * (1.0 + (u * u) / df).ln()).exp() };
    let abs_t = t.abs();
    let steps = 200usize;
    let h = abs_t / steps as f64;
    let mut integral = pdf(0.0) + pdf(abs_t);
    for i in 1..steps {
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        integral += coeff * pdf(i as f64 * h);
    }
    let area = (integral * h / 3.0).clamp(0.0, 0.5);
    if t > 0.0 {
        Ok(0.5 + area)
    } else {
        Ok(0.5 - area)
    }
}

/// Chi-square survival function $P(\chi^2_k \ge x)$ via regularized upper incomplete gamma.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64, StatsError> {
    if df <= 0.0 || !df.is_finite() {
        return Err(StatsError::InvalidParameter(
            "degrees of freedom must be positive".to_string(),
        ));
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    let a = 0.5 * df;
    let z = 0.5 * x;
    if z < a + 1.0 {
        // Series representation for lower incomplete gamma P(a, z)
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= z / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        let p_lower = sum * (-z + a * z.ln() - ln_gamma(a)).exp();
        Ok((1.0 - p_lower).clamp(0.0, 1.0))
    } else {
        // Lentz continued fraction for upper incomplete gamma Q(a, z)
        let mut b = z + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        Ok((h * (-z + a * z.ln() - ln_gamma(a)).exp()).clamp(0.0, 1.0))
    }
}

fn invert_symmetric_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, StatsError> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        if a[i].len() != n {
            return Err(StatsError::DimensionMismatch(
                "matrix must be square".to_string(),
            ));
        }
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(StatsError::SingularMatrix);
        }
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        for val in &mut aug[col] {
            *val /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    Ok(aug.into_iter().map(|r| r[n..].to_vec()).collect())
}

fn weighted_ridge_fit(
    y: &[f64],
    x_rows: &[Vec<f64>],
    weights: Option<&[f64]>,
    ridge_lambda: f64,
    hac_lags: Option<usize>,
) -> Result<RegressionResult, StatsError> {
    let n = y.len();
    let k = x_rows.first().map_or(0, Vec::len);
    if n <= k || k == 0 {
        return Err(StatsError::InsufficientData {
            required: k + 1,
            actual: n,
        });
    }
    if x_rows.len() != n || x_rows.iter().any(|r| r.len() != k) {
        return Err(StatsError::DimensionMismatch(
            "X row count or column count mismatch".to_string(),
        ));
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(StatsError::DimensionMismatch(
            "weights length must match y length".to_string(),
        ));
    }

    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for (row_idx, x_row) in x_rows.iter().enumerate() {
        let w = weights.map_or(1.0, |ws| ws[row_idx].max(0.0));
        for i in 0..k {
            xty[i] += w * x_row[i] * y[row_idx];
            for j in 0..k {
                xtx[i][j] += w * x_row[i] * x_row[j];
            }
        }
    }
    if ridge_lambda > 0.0 {
        for i in 0..k {
            xtx[i][i] += ridge_lambda;
        }
    }

    let xtx_inv = invert_symmetric_matrix(&xtx)?;
    let mut beta = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            beta[i] += xtx_inv[i][j] * xty[j];
        }
    }

    let residuals: Vec<f64> = (0..n)
        .map(|i| {
            let y_hat: f64 = (0..k).map(|j| x_rows[i][j] * beta[j]).sum();
            y[i] - y_hat
        })
        .collect();

    let mean_y = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let rss: f64 = residuals.iter().map(|&e| e * e).sum();
    let r_squared = if tss > 1e-12 {
        (1.0 - rss / tss).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let std_errors: Vec<f64> = if let Some(lags) = hac_lags {
        let mut omega = vec![vec![0.0; k]; k];
        for t in 0..n {
            for i in 0..k {
                for j in 0..k {
                    omega[i][j] += residuals[t] * residuals[t] * x_rows[t][i] * x_rows[t][j];
                }
            }
        }
        let max_lag = lags.min(n.saturating_sub(1));
        for l in 1..=max_lag {
            let w_bartlett = 1.0 - (l as f64) / ((max_lag + 1) as f64);
            for t in l..n {
                for i in 0..k {
                    for j in 0..k {
                        let cross = residuals[t]
                            * residuals[t - l]
                            * (x_rows[t][i] * x_rows[t - l][j] + x_rows[t - l][i] * x_rows[t][j]);
                        omega[i][j] += w_bartlett * cross;
                    }
                }
            }
        }
        let mut temp = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for m in 0..k {
                    temp[i][j] += xtx_inv[i][m] * omega[m][j];
                }
            }
        }
        let mut cov = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for m in 0..k {
                    cov[i][j] += temp[i][m] * xtx_inv[m][j];
                }
            }
        }
        (0..k).map(|i| cov[i][i].max(1e-18).sqrt()).collect()
    } else {
        let sigma2 = rss / ((n - k) as f64);
        (0..k)
            .map(|i| (sigma2 * xtx_inv[i][i]).max(1e-18).sqrt())
            .collect()
    };

    let t_stats: Vec<f64> = (0..k).map(|i| beta[i] / std_errors[i]).collect();

    Ok(RegressionResult {
        beta,
        std_errors,
        t_stats,
        r_squared,
        residuals,
    })
}

/// Ordinary Least Squares regression.
pub fn ols(y: &[f64], x_rows: &[Vec<f64>]) -> Result<RegressionResult, StatsError> {
    weighted_ridge_fit(y, x_rows, None, 0.0, None)
}

/// Weighted Least Squares regression.
pub fn wls(
    y: &[f64],
    x_rows: &[Vec<f64>],
    weights: &[f64],
) -> Result<RegressionResult, StatsError> {
    weighted_ridge_fit(y, x_rows, Some(weights), 0.0, None)
}

/// L2-regularized Ridge regression.
pub fn ridge_regression(
    y: &[f64],
    x_rows: &[Vec<f64>],
    lambda: f64,
) -> Result<RegressionResult, StatsError> {
    if lambda < 0.0 {
        return Err(StatsError::InvalidParameter(
            "ridge lambda must be non-negative".to_string(),
        ));
    }
    weighted_ridge_fit(y, x_rows, None, lambda, None)
}

/// OLS with Newey-West heteroskedasticity & autocorrelation consistent (HAC) standard errors.
pub fn newey_west_hac(
    y: &[f64],
    x_rows: &[Vec<f64>],
    lags: usize,
) -> Result<RegressionResult, StatsError> {
    weighted_ridge_fit(y, x_rows, None, 0.0, Some(lags))
}

/// Robust Huber regression via Iteratively Reweighted Least Squares (IRLS).
pub fn huber_irls_regression(
    y: &[f64],
    x_rows: &[Vec<f64>],
    delta: f64,
    max_iters: usize,
) -> Result<RegressionResult, StatsError> {
    if delta <= 0.0 {
        return Err(StatsError::InvalidParameter(
            "huber delta must be positive".to_string(),
        ));
    }
    let n = y.len();
    let mut weights = vec![1.0_f64; n];
    let mut fit = wls(y, x_rows, &weights)?;
    for _ in 0..max_iters.max(1) {
        for (i, &r) in fit.residuals.iter().enumerate() {
            let ar = r.abs();
            weights[i] = if ar <= delta { 1.0 } else { delta / ar };
        }
        fit = wls(y, x_rows, &weights)?;
    }
    Ok(fit)
}

/// Jarque-Bera normality test statistic $JB = \frac{n}{6}(S^2 + \frac{(K - 3)^2}{4})$.
pub fn jarque_bera_stat(series: &[f64]) -> Result<f64, StatsError> {
    let n = series.len();
    if n < 8 {
        return Err(StatsError::InsufficientData {
            required: 8,
            actual: n,
        });
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let m2 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if m2 <= 1e-18 {
        return Ok(0.0);
    }
    let m3 = series.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n as f64;
    let m4 = series.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n as f64;
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2);
    Ok((n as f64 / 6.0) * (skew * skew + 0.25 * (kurt - 3.0).powi(2)))
}

/// Augmented Dickey-Fuller (ADF) unit-root t-statistic on `series` with `lags` difference lags.
pub fn adf_statistic(series: &[f64], lags: usize) -> Result<f64, StatsError> {
    let min_len = lags + 8;
    if series.len() < min_len {
        return Err(StatsError::InsufficientData {
            required: min_len,
            actual: series.len(),
        });
    }
    let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let start = lags;
    let mut y = Vec::with_capacity(diff.len() - start);
    let mut x_rows = Vec::with_capacity(diff.len() - start);
    for t in start..diff.len() {
        y.push(diff[t]);
        let mut row = Vec::with_capacity(2 + lags);
        row.push(1.0);
        row.push(series[t]);
        for l in 1..=lags {
            row.push(diff[t - l]);
        }
        x_rows.push(row);
    }
    let res = ols(&y, &x_rows)?;
    Ok(res.t_stats[1])
}

/// KPSS stationarity test statistic (null hypothesis: level stationarity).
pub fn kpss_statistic(series: &[f64], lags: usize) -> Result<f64, StatsError> {
    let n = series.len();
    if n < 8 {
        return Err(StatsError::InsufficientData {
            required: 8,
            actual: n,
        });
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let resid: Vec<f64> = series.iter().map(|&v| v - mean).collect();
    let mut partial = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &e in &resid {
        acc += e;
        partial.push(acc);
    }
    let gamma0 = resid.iter().map(|&e| e * e).sum::<f64>() / n as f64;
    let mut long_run_var = gamma0;
    let max_lag = lags.min(n - 1);
    for l in 1..=max_lag {
        let w = 1.0 - (l as f64) / ((max_lag + 1) as f64);
        let cov: f64 = (l..n).map(|t| resid[t] * resid[t - l]).sum::<f64>() / n as f64;
        long_run_var += 2.0 * w * cov;
    }
    let denom = long_run_var.max(1e-14) * (n * n) as f64;
    Ok(partial.iter().map(|&s| s * s).sum::<f64>() / denom)
}

/// Ljung-Box Q autocorrelation portmanteau statistic across `1..=lags`.
pub fn ljung_box_q(series: &[f64], lags: usize) -> Result<f64, StatsError> {
    let n = series.len();
    if n <= lags + 2 || lags == 0 {
        return Err(StatsError::InsufficientData {
            required: lags + 3,
            actual: n,
        });
    }
    let mean = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|&v| (v - mean).powi(2)).sum::<f64>();
    if var <= 1e-18 {
        return Ok(0.0);
    }
    let mut q = 0.0;
    for k in 1..=lags {
        let num: f64 = (k..n)
            .map(|t| (series[t] - mean) * (series[t - k] - mean))
            .sum();
        let rho = num / var;
        q += (rho * rho) / ((n - k) as f64);
    }
    Ok((n as f64) * ((n + 2) as f64) * q)
}

/// Rescaled-range (R/S) Hurst exponent estimator.
pub fn hurst_rs(series: &[f64]) -> Result<f64, StatsError> {
    let n = series.len();
    if n < 16 {
        return Err(StatsError::InsufficientData {
            required: 16,
            actual: n,
        });
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut size = 8usize;
    while size <= n {
        let blocks = n / size;
        if blocks == 0 {
            break;
        }
        let mut rs_sum = 0.0;
        let mut valid = 0usize;
        for b in 0..blocks {
            let chunk = &series[(b * size)..((b + 1) * size)];
            let mean = chunk.iter().sum::<f64>() / size as f64;
            let std = (chunk.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / size as f64).sqrt();
            if std > 1e-12 {
                let mut dev = 0.0;
                let mut min_d = 0.0_f64;
                let mut max_d = 0.0_f64;
                for &v in chunk {
                    dev += v - mean;
                    min_d = min_d.min(dev);
                    max_d = max_d.max(dev);
                }
                rs_sum += (max_d - min_d) / std;
                valid += 1;
            }
        }
        if valid > 0 {
            xs.push((size as f64).ln());
            ys.push((rs_sum / valid as f64).ln());
        }
        size *= 2;
    }
    if xs.len() < 2 {
        return Ok(0.5);
    }
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let num: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
        .sum();
    let den: f64 = xs.iter().map(|&x| (x - x_mean).powi(2)).sum();
    Ok((num / den.max(1e-12)).clamp(0.0, 1.5))
}

/// Engle-Granger two-step cointegration test: returns `(cointegrating_slope, residual_adf_t_stat)`.
pub fn engle_granger_cointegration(
    y: &[f64],
    x: &[f64],
    adf_lags: usize,
) -> Result<(f64, f64), StatsError> {
    if y.len() != x.len() {
        return Err(StatsError::DimensionMismatch(
            "series y and x must have equal length".to_string(),
        ));
    }
    let x_rows: Vec<Vec<f64>> = x.iter().map(|&xi| vec![1.0, xi]).collect();
    let fit = ols(y, &x_rows)?;
    let adf_t = adf_statistic(&fit.residuals, adf_lags)?;
    Ok((fit.beta[1], adf_t))
}
