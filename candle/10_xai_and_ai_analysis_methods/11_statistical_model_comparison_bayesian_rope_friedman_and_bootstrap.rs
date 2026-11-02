//! Statistical Model Comparison Bayesian RoPE Friedman and Bootstrap
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// High-precision Abramowitz & Stegun erf approximation (error < 2e-7).
pub fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x_abs);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    sign * (1.0 - poly * (-x_abs * x_abs).exp())
}

/// McNemar's test with continuity correction (Dietterich, 1998).
pub fn mcnemar_test_statistic(b: &Tensor, c: &Tensor) -> Result<Tensor> {
    let b_val = b.to_scalar::<f32>()?;
    let c_val = c.to_scalar::<f32>()?;
    let diff = (b_val - c_val).abs() - 1.0f32;
    let stat = (diff * diff) / (b_val + c_val + 1e-8);
    Tensor::new(stat, b.device())
}

/// Nadeau-Bengio corrected resampled t-test for cross-validation (Nadeau & Bengio, 2003).
pub fn nadeau_bengio_corrected_t(
    diffs: &Tensor,
    n_train: f32,
    n_test: f32,
) -> Result<(Tensor, Tensor)> {
    let d = diffs.to_vec1::<f32>()?;
    let k = d.len() as f32;
    let d_bar: f32 = d.iter().sum::<f32>() / k;
    let s_sq: f32 = d.iter().map(|&x| (x - d_bar) * (x - d_bar)).sum::<f32>() / (k - 1.0);
    let c_factor = 1.0 / k + n_test / n_train;
    let sigma = (c_factor * s_sq + 1e-8).sqrt();
    let t_stat = d_bar / sigma;
    Ok((
        Tensor::new(t_stat, diffs.device())?,
        Tensor::new(sigma, diffs.device())?,
    ))
}

/// Friedman non-parametric multi-classifier comparison test (Demšar, 2006).
pub fn friedman_rank_statistic(ranks: &Tensor) -> Result<(Tensor, Tensor)> {
    let r = ranks.to_vec2::<f32>()?;
    let n = r.len() as f32;
    let k = r[0].len() as f32;

    let mut avg_ranks = vec![0.0f32; k as usize];
    for row in &r {
        for (j, &val) in row.iter().enumerate() {
            avg_ranks[j] += val;
        }
    }
    for val in &mut avg_ranks {
        *val /= n;
    }

    let sum_sq: f32 = avg_ranks.iter().map(|&x| x * x).sum();
    let chi_f = (12.0 * n / (k * (k + 1.0))) * (sum_sq - (k * (k + 1.0) * (k + 1.0)) / 4.0);

    Ok((
        Tensor::new(chi_f, ranks.device())?,
        Tensor::new(avg_ranks, ranks.device())?,
    ))
}

/// Bayesian correlated t-test with Region of Practical Equivalence (ROPE) (Benavoli et al., 2017).
pub fn bayesian_rope_probabilities(
    diffs: &Tensor,
    n_train: f32,
    n_test: f32,
    rope_radius: f32,
) -> Result<Tensor> {
    let d = diffs.to_vec1::<f32>()?;
    let k = d.len() as f32;
    let d_bar: f32 = d.iter().sum::<f32>() / k;
    let s_sq: f32 = d.iter().map(|&x| (x - d_bar) * (x - d_bar)).sum::<f32>() / (k - 1.0);
    let c_factor = 1.0 / k + n_test / n_train;
    let sigma = (c_factor * s_sq + 1e-8).sqrt();

    let r = rope_radius;
    let z1 = (r - d_bar) / sigma;
    let z2 = (-r - d_bar) / sigma;

    let sqrt2 = std::f32::consts::SQRT_2;
    let phi1 = 0.5f32 * (1.0f32 + erf_approx(z1 / sqrt2));
    let phi2 = 0.5f32 * (1.0f32 + erf_approx(z2 / sqrt2));

    let p_rope = phi1 - phi2;
    let p_better = 1.0f32 - phi1;
    let p_worse = phi2;

    Tensor::new(&[p_better, p_rope, p_worse], diffs.device())
}
