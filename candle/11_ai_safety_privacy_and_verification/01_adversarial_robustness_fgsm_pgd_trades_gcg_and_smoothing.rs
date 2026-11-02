//! Adversarial Robustness FGSM PGD TRADES GCG and Smoothing
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Fast Gradient Sign Method (Goodfellow et al., 2014).
pub fn fgsm_perturbation(
    x: &Tensor,
    grad: &Tensor,
    eps: f32,
    x_min: f32,
    x_max: f32,
) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let g_vec = grad.to_vec2::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            let sign = if g_vec[i][j] > 0.0 {
                1.0
            } else if g_vec[i][j] < 0.0 {
                -1.0
            } else {
                0.0
            };
            let val = x_vec[i][j] + eps * sign;
            out[i][j] = val.max(x_min).min(x_max);
        }
    }
    Tensor::new(out, x.device())
}

/// Projected Gradient Descent single step in L_infinity ball (Madry et al., 2018).
pub fn pgd_step(
    x_adv: &Tensor,
    x_orig: &Tensor,
    grad: &Tensor,
    alpha: f32,
    eps: f32,
    x_min: f32,
    x_max: f32,
) -> Result<Tensor> {
    let adv = x_adv.to_vec2::<f32>()?;
    let orig = x_orig.to_vec2::<f32>()?;
    let g_vec = grad.to_vec2::<f32>()?;
    let b = adv.len();
    let d = adv[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        for j in 0..d {
            let sign = if g_vec[i][j] > 0.0 {
                1.0
            } else if g_vec[i][j] < 0.0 {
                -1.0
            } else {
                0.0
            };
            let step = adv[i][j] + alpha * sign;
            let proj = step.max(orig[i][j] - eps).min(orig[i][j] + eps);
            out[i][j] = proj.max(x_min).min(x_max);
        }
    }
    Tensor::new(out, x_adv.device())
}

/// TRADES adversarial training regularized loss (Zhang et al., 2019).
pub fn trades_loss(
    clean_logits: &Tensor,
    adv_logits: &Tensor,
    labels: &Tensor,
    beta: f32,
) -> Result<Tensor> {
    let cl = clean_logits.to_vec2::<f32>()?;
    let al = adv_logits.to_vec2::<f32>()?;
    let lbls = labels.to_vec1::<f32>()?;
    let b = cl.len();
    let c = cl[0].len();

    let mut ce_sum = 0.0f32;
    let mut kl_sum = 0.0f32;

    for i in 0..b {
        let mut max_c = f32::NEG_INFINITY;
        let mut max_a = f32::NEG_INFINITY;
        for j in 0..c {
            if cl[i][j] > max_c {
                max_c = cl[i][j];
            }
            if al[i][j] > max_a {
                max_a = al[i][j];
            }
        }

        let mut sum_exp_c = 0.0f32;
        let mut sum_exp_a = 0.0f32;
        for j in 0..c {
            sum_exp_c += (cl[i][j] - max_c).exp();
            sum_exp_a += (al[i][j] - max_a).exp();
        }
        let lse_c = max_c + sum_exp_c.ln();
        let lse_a = max_a + sum_exp_a.ln();

        let y = lbls[i] as usize;
        ce_sum += lse_c - cl[i][y];

        let mut row_kl = 0.0f32;
        for j in 0..c {
            let p_c = (cl[i][j] - lse_c).exp();
            let log_p_c = cl[i][j] - lse_c;
            let log_p_a = al[i][j] - lse_a;
            row_kl += p_c * (log_p_c - log_p_a);
        }
        kl_sum += row_kl;
    }

    let loss = (ce_sum + beta * kl_sum) / (b as f32);
    Tensor::new(loss, clean_logits.device())
}

/// Normal quantile function (probit) via Acklam's rational approximation.
pub fn probit(p: f32) -> f32 {
    let p_clamped = p.max(1e-7).min(1.0 - 1e-7);
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p_clamped < p_low {
        let q = (-2.0 * p_clamped.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p_clamped <= p_high {
        let q = p_clamped - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p_clamped).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Certified L2 robustness radius via Randomized Smoothing (Cohen et al., 2019).
pub fn randomized_smoothing_radius(
    p_top: &Tensor,
    p_runnerup: &Tensor,
    sigma: f32,
) -> Result<Tensor> {
    let top = p_top.to_vec1::<f32>()?;
    let run = p_runnerup.to_vec1::<f32>()?;
    let n = top.len();

    let mut radii = vec![0.0f32; n];
    for i in 0..n {
        let probit_top = probit(top[i]);
        let probit_run = probit(run[i]);
        radii[i] = (sigma / 2.0) * (probit_top - probit_run);
    }
    Tensor::new(radii, p_top.device())
}
