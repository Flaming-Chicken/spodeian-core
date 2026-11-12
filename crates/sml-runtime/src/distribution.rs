//! General Distribution Transformer and Mapping Engine.
//!
//! Provides a mathematically rigorous, unified distribution framework that transforms
//! scalar floats, slices, and Candle tensors across:
//! 1. **Probability Density Function (PDF)**: $p(x) \ge 0$
//! 2. **Log-Density (Log-PDF)**: $\ln p(x)$ for stable maximum likelihood estimation
//! 3. **Cumulative Distribution Function (CDF)**: $F(x) = P(X \le x) \in [0, 1]$ (Probability Integral Transform)
//! 4. **Quantile / Inverse CDF**: $Q(u) = F^{-1}(u)$ for $u \in [0, 1]$
//! 5. **Discrete Probability Mass Function (PMF)**: Binned lattice mass $P(X \in [k h, (k+1) h)) = F((k+1) h) - F(k h)$
//! 6. **Discrete Lattice Quantization**: $k = \lfloor x / h \rfloor \in \mathbb{Z}$
//! 7. **Umbral Rational Decomposition**: $x = h \cdot (N + P/Q)$ mapping continuous reals directly
//!    to discrete integer coordinates $N \in \mathbb{Z}$ with sub-grid rational residue tracking.

use candle_core::{DType, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::umbral::RationalResidue;

/// Canonical distribution families supported for continuous and discrete mapping.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistributionKind {
    /// Normal distribution: $\mathcal{N}(\mu, \sigma^2)$
    Gaussian { mean: f64, std_dev: f64 },
    /// Student's t-distribution (heavy-tailed): $\mathrm{StudentT}(\nu, s)$
    StudentT { df: f64, scale: f64 },
    /// Logistic distribution: $\mathrm{Logistic}(\mu, s)$
    Logistic { location: f64, scale: f64 },
    /// Cauchy distribution (heavy-tailed): $\mathrm{Cauchy}(x_0, \gamma)$
    Cauchy { location: f64, scale: f64 },
    /// Uniform distribution: $\mathrm{Uniform}(a, b)$
    Uniform { lower: f64, upper: f64 },
    /// Laplace (double exponential) distribution: $\mathrm{Laplace}(\mu, b)$
    Laplace { location: f64, scale: f64 },
    /// Exponential distribution: $\mathrm{Exp}(\lambda)$ for $x \ge 0$
    Exponential { rate: f64 },
    /// Half-Normal distribution on $[0, \infty)$
    HalfNormal { scale: f64 },
}

/// Transformation target mode specifying how the float is modified.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistributionMapMode {
    /// Probability Density Function: $y = p(x)$
    Pdf,
    /// Natural Logarithm of PDF: $y = \ln p(x)$
    LogPdf,
    /// Cumulative Distribution Function: $y = F(x) \in [0, 1]$
    Cdf,
    /// Quantile / Inverse CDF: $x = F^{-1}(u)$ for $u \in [0, 1]$
    InverseCdf,
    /// Discrete Probability Mass Function for bin $[k h, (k+1) h)$: $\Delta_h F(k) = F((k+1) h) - F(k h)$
    DiscretePmf { grid_step: f64 },
    /// Discrete Lattice Coordinate: $k = \lfloor x / h \rfloor$
    DiscreteQuantize { grid_step: f64 },
    /// Umbral Rational Decomposition: $(N, P/Q)$ where $x \approx h(N + P/Q)$
    UmbralDecomposition { grid_step: f64, residue_denominator: i32 },
}

/// General distribution transformer capable of scalar, slice, and tensor mapping.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeneralDistribution {
    pub kind: DistributionKind,
}

impl GeneralDistribution {
    pub const fn new(kind: DistributionKind) -> Self {
        Self { kind }
    }

    pub const fn standard_gaussian() -> Self {
        Self {
            kind: DistributionKind::Gaussian {
                mean: 0.0,
                std_dev: 1.0,
            },
        }
    }

    pub const fn standard_logistic() -> Self {
        Self {
            kind: DistributionKind::Logistic {
                location: 0.0,
                scale: 1.0,
            },
        }
    }

    pub const fn standard_cauchy() -> Self {
        Self {
            kind: DistributionKind::Cauchy {
                location: 0.0,
                scale: 1.0,
            },
        }
    }

    pub const fn uniform(lower: f64, upper: f64) -> Self {
        Self {
            kind: DistributionKind::Uniform { lower, upper },
        }
    }

    pub const fn student_t(df: f64, scale: f64) -> Self {
        Self {
            kind: DistributionKind::StudentT { df, scale },
        }
    }

    pub const fn laplace(location: f64, scale: f64) -> Self {
        Self {
            kind: DistributionKind::Laplace { location, scale },
        }
    }

    pub const fn exponential(rate: f64) -> Self {
        Self {
            kind: DistributionKind::Exponential { rate },
        }
    }

    pub const fn half_normal(scale: f64) -> Self {
        Self {
            kind: DistributionKind::HalfNormal { scale },
        }
    }

    // ========================================================================
    // Continuous Analytical Methods: PDF, Log-PDF, CDF, Inverse CDF
    // ========================================================================

    /// Evaluates the Cumulative Distribution Function $F(x) = P(X \le x) \in [0, 1]$.
    pub fn cdf(&self, x: f64) -> f64 {
        match self.kind {
            DistributionKind::Gaussian { mean, std_dev } => {
                let s = std_dev.max(1e-15);
                let z = (x - mean) / (s * std::f64::consts::SQRT_2);
                0.5 * (1.0 + erf_approx(z))
            }
            DistributionKind::StudentT { df, scale } => {
                let s = scale.max(1e-15);
                let z = (x / s).clamp(-100.0, 100.0);
                if df <= 1.05 {
                    0.5 + z.atan() / std::f64::consts::PI
                } else {
                    let eff_z = z / (df / (df - 2.0).max(1e-3)).sqrt().max(1e-6);
                    0.5 * (1.0 + erf_approx(eff_z / std::f64::consts::SQRT_2))
                }
            }
            DistributionKind::Logistic { location, scale } => {
                let s = scale.max(1e-15);
                let z = (x - location) / s;
                // Stable sigmoid computation
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                }
            }
            DistributionKind::Cauchy { location, scale } => {
                let s = scale.max(1e-15);
                0.5 + ((x - location) / s).atan() / std::f64::consts::PI
            }
            DistributionKind::Uniform { lower, upper } => {
                let diff = (upper - lower).max(1e-15);
                if x <= lower {
                    0.0
                } else if x >= upper {
                    1.0
                } else {
                    (x - lower) / diff
                }
            }
            DistributionKind::Laplace { location, scale } => {
                let b = scale.max(1e-15);
                let diff = x - location;
                if diff < 0.0 {
                    0.5 * (diff / b).exp()
                } else {
                    1.0 - 0.5 * (-diff / b).exp()
                }
            }
            DistributionKind::Exponential { rate } => {
                let r = rate.max(1e-15);
                if x <= 0.0 {
                    0.0
                } else {
                    1.0 - (-r * x).exp()
                }
            }
            DistributionKind::HalfNormal { scale } => {
                let s = scale.max(1e-15);
                if x <= 0.0 {
                    0.0
                } else {
                    let z = x / (s * std::f64::consts::SQRT_2);
                    erf_approx(z)
                }
            }
        }
    }

    /// Evaluates the Probability Density Function $p(x) \ge 0$.
    pub fn pdf(&self, x: f64) -> f64 {
        match self.kind {
            DistributionKind::Gaussian { mean, std_dev } => {
                let s = std_dev.max(1e-15);
                let z = (x - mean) / s;
                let norm = 1.0 / (s * (2.0 * std::f64::consts::PI).sqrt());
                norm * (-0.5 * z * z).exp()
            }
            DistributionKind::StudentT { df, scale } => {
                let s = scale.max(1e-15);
                let z = x / s;
                let nu = df.max(1.0);
                let base = 1.0 + (z * z) / nu;
                let norm = 1.0 / (s * (std::f64::consts::PI * nu).sqrt());
                norm * base.powf(-0.5 * (nu + 1.0))
            }
            DistributionKind::Logistic { location, scale } => {
                let s = scale.max(1e-15);
                let z = (x - location) / s;
                let ez = (-z.abs()).exp();
                let denom = 1.0 + ez;
                ez / (s * denom * denom)
            }
            DistributionKind::Cauchy { location, scale } => {
                let s = scale.max(1e-15);
                let z = (x - location) / s;
                1.0 / (std::f64::consts::PI * s * (1.0 + z * z))
            }
            DistributionKind::Uniform { lower, upper } => {
                let diff = (upper - lower).max(1e-15);
                if x >= lower && x <= upper {
                    1.0 / diff
                } else {
                    0.0
                }
            }
            DistributionKind::Laplace { location, scale } => {
                let b = scale.max(1e-15);
                let diff = (x - location).abs();
                (0.5 / b) * (-diff / b).exp()
            }
            DistributionKind::Exponential { rate } => {
                let r = rate.max(1e-15);
                if x < 0.0 {
                    0.0
                } else {
                    r * (-r * x).exp()
                }
            }
            DistributionKind::HalfNormal { scale } => {
                let s = scale.max(1e-15);
                if x < 0.0 {
                    0.0
                } else {
                    let z = x / s;
                    ((2.0 / std::f64::consts::PI).sqrt() / s) * (-0.5 * z * z).exp()
                }
            }
        }
    }

    /// Evaluates the Natural Logarithm of the Probability Density Function $\ln p(x)$.
    pub fn log_pdf(&self, x: f64) -> f64 {
        match self.kind {
            DistributionKind::Gaussian { mean, std_dev } => {
                let s = std_dev.max(1e-15);
                let z = (x - mean) / s;
                -s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * z * z
            }
            DistributionKind::StudentT { df, scale } => {
                let s = scale.max(1e-15);
                let z = x / s;
                let nu = df.max(1.0);
                -s.ln() - 0.5 * (std::f64::consts::PI * nu).ln() - 0.5 * (nu + 1.0) * (1.0 + (z * z) / nu).ln()
            }
            DistributionKind::Logistic { location, scale } => {
                let s = scale.max(1e-15);
                let z = (x - location) / s;
                let abs_z = z.abs();
                -s.ln() - abs_z - 2.0 * (1.0 + (-abs_z).exp()).ln()
            }
            DistributionKind::Cauchy { location, scale } => {
                let s = scale.max(1e-15);
                let z = (x - location) / s;
                -(std::f64::consts::PI * s).ln() - (1.0 + z * z).ln()
            }
            DistributionKind::Uniform { lower, upper } => {
                let diff = (upper - lower).max(1e-15);
                if x >= lower && x <= upper {
                    -diff.ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionKind::Laplace { location, scale } => {
                let b = scale.max(1e-15);
                let diff = (x - location).abs();
                -(2.0 * b).ln() - diff / b
            }
            DistributionKind::Exponential { rate } => {
                let r = rate.max(1e-15);
                if x < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    r.ln() - r * x
                }
            }
            DistributionKind::HalfNormal { scale } => {
                let s = scale.max(1e-15);
                if x < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    let z = x / s;
                    0.5 * (2.0 / std::f64::consts::PI).ln() - s.ln() - 0.5 * z * z
                }
            }
        }
    }

    /// Evaluates the Quantile / Inverse CDF $Q(u) = F^{-1}(u)$ for $u \in [0, 1]$.
    pub fn inverse_cdf(&self, u: f64) -> f64 {
        let u_clamped = u.clamp(1e-12, 1.0 - 1e-12);
        match self.kind {
            DistributionKind::Gaussian { mean, std_dev } => {
                let s = std_dev.max(1e-15);
                mean + s * std::f64::consts::SQRT_2 * inv_erf_approx(2.0 * u_clamped - 1.0)
            }
            DistributionKind::StudentT { df, scale } => {
                let s = scale.max(1e-15);
                let nu = df.max(1.0);
                if nu <= 1.05 {
                    s * (std::f64::consts::PI * (u_clamped - 0.5)).tan()
                } else {
                    let z = std::f64::consts::SQRT_2 * inv_erf_approx(2.0 * u_clamped - 1.0);
                    let z2 = z * z;
                    let t = z * (1.0 + (z2 + 1.0) / (4.0 * nu) + (5.0 * z2 * z2 + 16.0 * z2 + 3.0) / (96.0 * nu * nu));
                    s * t
                }
            }
            DistributionKind::Logistic { location, scale } => {
                let s = scale.max(1e-15);
                location + s * (u_clamped / (1.0 - u_clamped)).ln()
            }
            DistributionKind::Cauchy { location, scale } => {
                let s = scale.max(1e-15);
                location + s * (std::f64::consts::PI * (u_clamped - 0.5)).tan()
            }
            DistributionKind::Uniform { lower, upper } => {
                lower + u_clamped * (upper - lower)
            }
            DistributionKind::Laplace { location, scale } => {
                let b = scale.max(1e-15);
                if u_clamped < 0.5 {
                    location + b * (2.0 * u_clamped).ln()
                } else {
                    location - b * (2.0 * (1.0 - u_clamped)).ln()
                }
            }
            DistributionKind::Exponential { rate } => {
                let r = rate.max(1e-15);
                -(1.0 - u_clamped).ln() / r
            }
            DistributionKind::HalfNormal { scale } => {
                let s = scale.max(1e-15);
                s * std::f64::consts::SQRT_2 * inv_erf_approx(u_clamped)
            }
        }
    }

    // ========================================================================
    // Discrete & Umbral Lattice Equivalents
    // ========================================================================

    /// Evaluates the discrete Probability Mass Function (PMF) on lattice cell $[k h, (k+1) h)$.
    ///
    /// Corresponds to the discrete difference operator on the CDF:
    /// $$p_{\mathrm{discrete}}(k) = \Delta_h F(k) = F((k+1) h) - F(k h) = \int_{kh}^{(k+1)h} p(x) dx$$
    pub fn discrete_pmf(&self, k: i64, grid_step: f64) -> f64 {
        let h = grid_step.abs().max(1e-8);
        let x_low = k as f64 * h;
        let x_high = (k + 1) as f64 * h;
        let cdf_high = self.cdf(x_high);
        let cdf_low = self.cdf(x_low);
        (cdf_high - cdf_low).max(0.0)
    }

    /// Quantizes a continuous float $x$ to its discrete lattice coordinate $k = \lfloor x / h \rfloor$.
    pub fn discrete_quantize(&self, x: f64, grid_step: f64) -> i64 {
        let h = grid_step.abs().max(1e-8);
        (x / h).floor() as i64
    }

    /// Decomposes a continuous float $x$ into an exact integer lattice coordinate $N \in \mathbb{Z}$
    /// and a rational sub-grid residue $P/Q$, enabling zero-allocation umbral integer tracking.
    pub fn umbral_decompose(&self, x: f64, grid_step: f64, residue_denominator: i32) -> (i32, RationalResidue) {
        let h = grid_step.abs().max(1e-8);
        let q = residue_denominator.max(1);
        let scaled = x / h;
        let n = scaled.floor() as i32;
        let frac = scaled - (n as f64);
        let mut p = (frac * (q as f64)).round() as i32;
        let mut carried_n = n;
        if p >= q {
            carried_n += 1;
            p -= q;
        } else if p < 0 {
            carried_n -= 1;
            p += q;
        }

        let mut res = RationalResidue::new(q);
        res.numerator = p;
        (carried_n, res)
    }

    // ========================================================================
    // Unified Mapping Interface
    // ========================================================================

    /// Maps a single continuous scalar float according to the requested transformation mode.
    pub fn map_scalar(&self, x: f64, mode: DistributionMapMode) -> f64 {
        match mode {
            DistributionMapMode::Pdf => self.pdf(x),
            DistributionMapMode::LogPdf => self.log_pdf(x),
            DistributionMapMode::Cdf => self.cdf(x),
            DistributionMapMode::InverseCdf => self.inverse_cdf(x),
            DistributionMapMode::DiscretePmf { grid_step } => {
                let k = self.discrete_quantize(x, grid_step);
                self.discrete_pmf(k, grid_step)
            }
            DistributionMapMode::DiscreteQuantize { grid_step } => {
                let k = self.discrete_quantize(x, grid_step);
                k as f64 * grid_step
            }
            DistributionMapMode::UmbralDecomposition { grid_step, residue_denominator } => {
                let (n, res) = self.umbral_decompose(x, grid_step, residue_denominator);
                grid_step * (n as f64 + res.numerator as f64 / res.denominator as f64)
            }
        }
    }

    /// Maps a slice of floating-point values in place or into a target buffer.
    pub fn map_slice(&self, input: &[f64], output: &mut [f64], mode: DistributionMapMode) {
        let n = input.len().min(output.len());
        for i in 0..n {
            output[i] = self.map_scalar(input[i], mode);
        }
    }

    /// Maps a Candle tensor through the distribution transformation.
    pub fn map_tensor(&self, tensor: &Tensor, mode: DistributionMapMode) -> Result<Tensor> {
        let orig_dtype = tensor.dtype();
        let f64_tensor = tensor.to_dtype(DType::F64)?;
        let flat_f64 = f64_tensor.flatten_all()?.to_vec1::<f64>()?;
        let mut mapped_f64 = vec![0.0f64; flat_f64.len()];
        self.map_slice(&flat_f64, &mut mapped_f64, mode);

        let out = Tensor::from_vec(mapped_f64, tensor.shape(), tensor.device())?;
        out.to_dtype(orig_dtype)
    }
}

// ============================================================================
// Generalized Distribution Floats
// ============================================================================

/// Origin describing how the [`DistributionFloat`] codebook was constructed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionFloatSource {
    /// Parametric analytical distribution family.
    Parametric(DistributionKind),
    /// Empirical distribution calibrated from observed data samples.
    Empirical { num_samples: usize },
}

/// Generalized Distribution-Matched Floating-Point Quantizer.
///
/// Unlike standard IEEE floats (linear mantissa with logarithmic exponent)
/// or NF4 (which is hard-coded to standard normal quantiles), `DistributionFloat` constructs
/// optimal information-theoretic quantile codebooks tailored to *arbitrary* underlying continuous
/// or empirical distributions:
/// - Gaussian / Normal: $\mathcal{N}(\mu, \sigma^2)$ (matches NF4/NF8)
/// - Laplace (Double-Exponential): $\mathrm{Laplace}(\mu, b)$ (optimal for sparse / L1-regularized weights)
/// - Student-t / Cauchy: $\mathrm{StudentT}(\nu)$ / $\mathrm{Cauchy}(x_0, \gamma)$ (optimal for heavy-tailed activations & outlier-prone attention gradients)
/// - Logistic: $\mathrm{Logistic}(\mu, s)$ (optimal for sigmoidal output spaces)
/// - Uniform: $\mathrm{Uniform}(a, b)$ (optimal for bounded phase-space states)
/// - Exponential / Half-Normal: (optimal for non-negative post-ReLU/GELU activations)
/// - Empirical: (fitted directly to user-supplied calibration data tensors)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionFloat {
    pub name: String,
    pub static_name: &'static str,
    pub source: DistributionFloatSource,
    pub bit_width: usize,
    pub codebook: Vec<f64>,
    pub min_value: f64,
    pub max_value: f64,
}

impl DistributionFloat {
    /// Constructs a `DistributionFloat` from an analytical `DistributionKind` and bit-width $B \in [2, 8]$.
    pub fn from_distribution(static_name: &'static str, kind: DistributionKind, bit_width: usize) -> Self {
        let b = bit_width.clamp(2, 8);
        let k = 1usize << b;
        let dist = GeneralDistribution::new(kind);

        let codebook = match kind {
            DistributionKind::Exponential { .. } | DistributionKind::HalfNormal { .. } => {
                // One-sided distribution on [0, infinity)
                let mut cb = Vec::with_capacity(k);
                cb.push(0.0);
                for j in 1..k {
                    let p = (j as f64) / (k as f64);
                    let q = dist.inverse_cdf(p.min(0.9999));
                    cb.push(q);
                }
                let max_q = cb[k - 1].max(1e-12);
                for v in &mut cb {
                    *v /= max_q;
                }
                cb
            }
            _ => {
                // Symmetric zero-centered distribution: reserve exact 0.0
                let n_neg = (k / 2).saturating_sub(1).max(1);
                let n_pos = k / 2;
                let mut cb = Vec::with_capacity(k);

                // Negative quantiles in (0, 0.5)
                for i in 0..n_neg {
                    let p = ((i as f64 + 0.5) / (n_neg as f64 + 0.5)) * 0.5;
                    let q = dist.inverse_cdf(p.clamp(1e-6, 0.4999));
                    cb.push(q);
                }
                // Exact zero level
                cb.push(0.0);
                // Positive quantiles in (0.5, 1.0)
                for j in 0..n_pos {
                    let p = 0.5 + ((j as f64 + 0.5) / (n_pos as f64 + 0.5)) * 0.5;
                    let q = dist.inverse_cdf(p.clamp(0.5001, 0.9999));
                    cb.push(q);
                }

                // Normalize so endpoints map to [-1.0, 1.0]
                let max_abs = cb[0].abs().max(cb[cb.len() - 1].abs()).max(1e-12);
                for v in &mut cb {
                    *v /= max_abs;
                }
                cb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                cb
            }
        };

        let min_value = codebook[0];
        let max_value = codebook[codebook.len() - 1];

        Self {
            name: format!("{}_{}bit", static_name, b),
            static_name,
            source: DistributionFloatSource::Parametric(kind),
            bit_width: b,
            codebook,
            min_value,
            max_value,
        }
    }

    /// Constructs an empirical `DistributionFloat` fitted directly to user data samples.
    pub fn from_empirical_data(static_name: &'static str, data: &[f64], bit_width: usize) -> Self {
        let b = bit_width.clamp(2, 8);
        let k = 1usize << b;

        let mut valid_data: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
        if valid_data.is_empty() {
            return Self::gaussian(b);
        }
        valid_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let m = valid_data.len();
        let mut cb = Vec::with_capacity(k);
        for i in 0..k {
            let p = (i as f64 + 0.5) / (k as f64);
            let idx = (((m - 1) as f64 * p).round() as usize).min(m - 1);
            cb.push(valid_data[idx]);
        }

        cb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for i in 1..cb.len() {
            if cb[i] <= cb[i - 1] {
                cb[i] = cb[i - 1] + 1e-7;
            }
        }

        let max_abs = cb[0].abs().max(cb[cb.len() - 1].abs()).max(1e-12);
        for v in &mut cb {
            *v /= max_abs;
        }

        let min_value = cb[0];
        let max_value = cb[cb.len() - 1];

        Self {
            name: format!("{}_{}bit", static_name, b),
            static_name,
            source: DistributionFloatSource::Empirical { num_samples: m },
            bit_width: b,
            codebook: cb,
            min_value,
            max_value,
        }
    }

    pub fn gaussian(bit_width: usize) -> Self {
        Self::from_distribution(
            "DistributionFloat::Gaussian",
            DistributionKind::Gaussian { mean: 0.0, std_dev: 1.0 },
            bit_width,
        )
    }

    pub fn laplace(bit_width: usize, scale: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::Laplace",
            DistributionKind::Laplace { location: 0.0, scale },
            bit_width,
        )
    }

    pub fn student_t(bit_width: usize, df: f64, scale: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::StudentT",
            DistributionKind::StudentT { df, scale },
            bit_width,
        )
    }

    pub fn cauchy(bit_width: usize, scale: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::Cauchy",
            DistributionKind::Cauchy { location: 0.0, scale },
            bit_width,
        )
    }

    pub fn logistic(bit_width: usize, scale: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::Logistic",
            DistributionKind::Logistic { location: 0.0, scale },
            bit_width,
        )
    }

    pub fn uniform(bit_width: usize, lower: f64, upper: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::Uniform",
            DistributionKind::Uniform { lower, upper },
            bit_width,
        )
    }

    pub fn exponential(bit_width: usize, rate: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::Exponential",
            DistributionKind::Exponential { rate },
            bit_width,
        )
    }

    pub fn half_normal(bit_width: usize, scale: f64) -> Self {
        Self::from_distribution(
            "DistributionFloat::HalfNormal",
            DistributionKind::HalfNormal { scale },
            bit_width,
        )
    }

    /// Fast binary-search encoding returning the index of the nearest codebook quantile.
    #[inline]
    pub fn encode(&self, val: f64) -> usize {
        let n = self.codebook.len();
        if val <= self.codebook[0] {
            return 0;
        }
        if val >= self.codebook[n - 1] {
            return n - 1;
        }
        match self.codebook.binary_search_by(|v| v.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => idx,
            Err(idx) => {
                let left = self.codebook[idx - 1];
                let right = self.codebook[idx];
                if (val - left).abs() <= (right - val).abs() {
                    idx - 1
                } else {
                    idx
                }
            }
        }
    }

    /// Fast $O(1)$ table-lookup decoding.
    #[inline]
    pub fn decode(&self, idx: usize) -> f64 {
        self.codebook[idx.min(self.codebook.len() - 1)]
    }

    /// Direct scalar quantization to the nearest quantile level.
    #[inline]
    pub fn quantize(&self, val: f64) -> f64 {
        self.decode(self.encode(val))
    }

    /// Unbiased stochastic rounding: guarantees $\mathbb{E}[Q_{\text{stoch}}(x)] = x$.
    pub fn quantize_stochastic(&self, val: f64, rand_u: f64) -> f64 {
        let n = self.codebook.len();
        if val <= self.codebook[0] {
            return self.codebook[0];
        }
        if val >= self.codebook[n - 1] {
            return self.codebook[n - 1];
        }
        match self.codebook.binary_search_by(|v| v.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(idx) => self.codebook[idx],
            Err(idx) => {
                let left = self.codebook[idx - 1];
                let right = self.codebook[idx];
                let diff = right - left;
                if diff <= 1e-15 {
                    left
                } else {
                    let prob_right = (val - left) / diff;
                    if rand_u < prob_right {
                        right
                    } else {
                        left
                    }
                }
            }
        }
    }

    /// In-place slice quantization.
    pub fn quantize_slice(&self, slice: &mut [f64]) {
        for v in slice.iter_mut() {
            *v = self.quantize(*v);
        }
    }

    /// In-place slice compensated error-diffusion quantization carrying residue $r_i$ forward.
    /// Guarantees that the total sum error $|\sum \text{dst} - \sum \text{src}| \le \frac{1}{2}\Delta_{\max}$.
    pub fn quantize_slice_compensated(&self, src: &[f64], dst: &mut [f64]) {
        let n = src.len().min(dst.len());
        let mut residue = 0.0f64;
        for i in 0..n {
            let u = src[i] + residue;
            let q = self.quantize(u);
            residue = u - q;
            dst[i] = q;
        }
    }

    /// In-place slice stochastic quantization using a pseudo-random seed.
    pub fn quantize_slice_stochastic(&self, src: &[f64], dst: &mut [f64], seed: u64) {
        let n = src.len().min(dst.len());
        let mut rng = seed.max(1);
        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rand_u = ((rng >> 33) as f64) / ((1u64 << 31) as f64);
            dst[i] = self.quantize_stochastic(src[i], rand_u);
        }
    }

    /// Block-max normalized quantization ($s = \max |x_i|$).
    pub fn quantize_block_scaled(&self, src: &[f64], dst: &mut [f64]) -> f64 {
        let n = src.len().min(dst.len());
        let mut max_abs = 0.0f64;
        for &v in &src[..n] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs < 1e-12 {
            for d in &mut dst[..n] {
                *d = 0.0;
            }
            return 0.0;
        }
        let inv_scale = 1.0 / max_abs;
        for i in 0..n {
            let norm_x = src[i] * inv_scale;
            let q = self.quantize(norm_x);
            dst[i] = q * max_abs;
        }
        max_abs
    }

    /// Block-max normalized quantization with compensated error-diffusion for minimal drift.
    pub fn quantize_block_scaled_compensated(&self, src: &[f64], dst: &mut [f64]) -> f64 {
        let n = src.len().min(dst.len());
        let mut max_abs = 0.0f64;
        for &v in &src[..n] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs < 1e-12 {
            for d in &mut dst[..n] {
                *d = 0.0;
            }
            return 0.0;
        }
        let inv_scale = 1.0 / max_abs;
        let mut residue = 0.0f64;
        for i in 0..n {
            let u = src[i] * inv_scale + residue;
            let q = self.quantize(u);
            residue = u - q;
            dst[i] = q * max_abs;
        }
        max_abs
    }
}

impl crate::precision::NumericPrecision for DistributionFloat {
    fn name(&self) -> &'static str {
        self.static_name
    }

    fn bits(&self) -> usize {
        self.bit_width
    }

    fn is_distribution_float(&self) -> bool {
        true
    }

    fn is_quantile_normal(&self) -> bool {
        matches!(self.source, DistributionFloatSource::Parametric(DistributionKind::Gaussian { .. }))
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        let n = self.codebook.len();
        if n >= 2 {
            (self.max_value - self.min_value) / (n as f64 - 1.0)
        } else {
            0.1
        }
    }

    fn denominator_guard(&self) -> f64 {
        self.machine_epsilon(None) * 0.1
    }

    fn temporal_guard(&self) -> f64 {
        self.machine_epsilon(None) * 0.5
    }

    fn energy_drift_guard(&self) -> f64 {
        self.machine_epsilon(None) * 0.2
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.05
    }

    fn default_tolerances(&self) -> (f64, f64) {
        let eps = self.machine_epsilon(None);
        (eps, eps)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        3.0 * base
    }

    fn quantize(&self, val: f64) -> f64 {
        self.quantize(val)
    }

    fn quantize_slice(&self, slice: &mut [f64]) {
        self.quantize_slice(slice);
    }
}

// ============================================================================
// Numerical Helper Functions: High-Accuracy erf and inv_erf
// ============================================================================

/// Abramowitz and Stegun Formula 7.1.26 polynomial approximation for erf(x).
/// Maximum absolute error |epsilon| < 1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();

    // Constant coefficients
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x_abs);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let result = 1.0 - poly * (-x_abs * x_abs).exp();

    sign * result
}

/// Winitzki approximation for the inverse error function erf^{-1}(x).
fn inv_erf_approx(x: f64) -> f64 {
    let x_clamped = x.clamp(-0.9999999, 0.9999999);
    let a = 0.147;
    let one_minus_x_sq = 1.0 - x_clamped * x_clamped;
    let ln_term = one_minus_x_sq.ln();
    let term1 = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let inner_sqrt = term1 * term1 - ln_term / a;
    let sign = if x_clamped < 0.0 { -1.0 } else { 1.0 };

    sign * ((inner_sqrt.sqrt() - term1).max(0.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_pdf_cdf_roundtrip() {
        let dist = GeneralDistribution::standard_gaussian();
        // At 0: PDF is 1/sqrt(2pi) approx 0.39894, CDF is 0.5
        let pdf_0 = dist.map_scalar(0.0, DistributionMapMode::Pdf);
        let cdf_0 = dist.map_scalar(0.0, DistributionMapMode::Cdf);
        assert!((pdf_0 - 0.39894228).abs() < 1e-4);
        assert!((cdf_0 - 0.5).abs() < 1e-4);

        // Inverse CDF at 0.5 is 0.0
        let inv_05 = dist.map_scalar(0.5, DistributionMapMode::InverseCdf);
        assert!(inv_05.abs() < 1e-4);
    }

    #[test]
    fn test_logistic_cdf_and_discrete_pmf() {
        let dist = GeneralDistribution::standard_logistic();
        // Sum of discrete PMF over all bins must equal 1.0
        let grid_step = 0.5;
        let mut sum_pmf = 0.0;
        for k in -50..50 {
            sum_pmf += dist.discrete_pmf(k, grid_step);
        }
        assert!((sum_pmf - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_umbral_decomposition_and_reconstruction() {
        let dist = GeneralDistribution::standard_gaussian();
        let x = std::f64::consts::PI;
        let h = 0.25;
        let q = 1024;
        let (n, res) = dist.umbral_decompose(x, h, q);

        let reconstructed = h * (n as f64 + (res.numerator as f64 / res.denominator as f64));
        assert!((reconstructed - x).abs() < (h / q as f64));
    }

    #[test]
    fn test_distribution_float_codebook_and_quantization() {
        // Test Gaussian 4-bit (16 levels)
        let df_gauss = DistributionFloat::gaussian(4);
        assert_eq!(df_gauss.codebook.len(), 16);
        assert!(df_gauss.codebook.windows(2).all(|w| w[0] < w[1]));
        assert!(df_gauss.codebook.iter().any(|&v| v.abs() < 1e-12));

        // Test Student-t 3-bit (8 levels)
        let df_t = DistributionFloat::student_t(3, 4.0, 1.0);
        assert_eq!(df_t.codebook.len(), 8);
        assert!(df_t.codebook.windows(2).all(|w| w[0] < w[1]));

        // Test Laplace 4-bit (16 levels)
        let df_laplace = DistributionFloat::laplace(4, 1.0);
        assert_eq!(df_laplace.codebook.len(), 16);
        assert!(df_laplace.codebook.windows(2).all(|w| w[0] < w[1]));

        // Quantize scalar roundtrip
        let q = df_gauss.quantize(0.0);
        assert!(q.abs() < 1e-12);
    }

    #[test]
    fn test_distribution_float_empirical() {
        let mut data = Vec::with_capacity(1000);
        for i in 0..1000 {
            let u = (i as f64) / 1000.0;
            data.push((u * 2.0 - 1.0).powi(3)); // cubic distribution
        }
        let df_emp = DistributionFloat::from_empirical_data("cubic_4bit", &data, 4);
        assert_eq!(df_emp.codebook.len(), 16);
        assert!(df_emp.codebook.windows(2).all(|w| w[0] < w[1]));
        let q = df_emp.quantize(0.5);
        assert!(q.is_finite());
    }

    #[test]
    fn test_distribution_float_compensated_drift_suppression() {
        let df = DistributionFloat::student_t(4, 3.0, 1.0);
        let n = 2000;
        let mut src = Vec::with_capacity(n);
        let mut rng = 42u64;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((rng >> 33) as f64) / ((1u64 << 31) as f64);
            // Non-symmetric offset [0.1, 0.9]
            src.push(0.1 + 0.8 * u);
        }

        let mut nearest_dst = vec![0.0f64; n];
        let mut comp_dst = vec![0.0f64; n];

        // Standard nearest
        for i in 0..n {
            nearest_dst[i] = df.quantize(src[i]);
        }

        // Compensated error diffusion
        df.quantize_slice_compensated(&src, &mut comp_dst);

        let orig_sum: f64 = src.iter().sum();
        let nearest_sum: f64 = nearest_dst.iter().sum();
        let comp_sum: f64 = comp_dst.iter().sum();

        let nearest_drift = (nearest_sum - orig_sum).abs();
        let comp_drift = (comp_sum - orig_sum).abs();

        // Compensated drift must be bounded by half step size (<= 0.5)
        assert!(comp_drift <= 0.5, "Compensated drift {} exceeded 0.5", comp_drift);
        assert!(
            comp_drift < nearest_drift * 0.1,
            "Compensated drift {} was not >= 10x smaller than nearest drift {}",
            comp_drift,
            nearest_drift
        );
    }

    #[test]
    fn test_distribution_float_stochastic_expectation() {
        let df = DistributionFloat::gaussian(4);
        let target = 0.2345;
        let trials = 20_000;
        let mut sum = 0.0f64;
        let mut rng = 99999u64;
        for _ in 0..trials {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rand_u = ((rng >> 33) as f64) / ((1u64 << 31) as f64);
            sum += df.quantize_stochastic(target, rand_u);
        }
        let empirical_mean = sum / (trials as f64);
        assert!(
            (empirical_mean - target).abs() < 0.015,
            "Empirical mean {} deviated from target {}",
            empirical_mean,
            target
        );
    }
}
