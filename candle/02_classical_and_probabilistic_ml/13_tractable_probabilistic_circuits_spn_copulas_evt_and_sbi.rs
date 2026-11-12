//! Tractable Probabilistic Circuits (SPN), Copulas, Extreme Value Theory (EVT), and SBI in Candle.
//!
//! Features:
//! - Tractable Sum-Product Networks (SPN) with smoothness and decomposability
//! - Archimedean Clayton Copula modeling tail dependence
//! - Extreme Value Theory (EVT) Generalized Pareto Distribution (GPD)
//! - Simulation-Based Inference (SBI) neural density ratio estimation

use candle_core::{Result, Tensor};

/// Evaluates a smooth and decomposable 4-variable Sum-Product Network (SPN).
pub fn evaluate_spn(
    leaves_p: &Tensor,
    leaf_evidence: &Tensor,
    prod_weights: &Tensor,
) -> Result<Tensor> {
    let leaves_vec: Vec<Vec<f32>> = leaves_p.to_vec2()?;
    let ev_vec: Vec<f32> = leaf_evidence.to_vec1()?;
    let weights_vec: Vec<f32> = prod_weights.to_vec1()?;

    let mut leaf_vals = vec![vec![1.0f32; 2]; 4];
    for i in 0..4 {
        let ev = ev_vec[i];
        if (ev - 1.0f32).abs() < 1e-5 {
            leaf_vals[i][0] = leaves_vec[i][0];
            leaf_vals[i][1] = leaves_vec[i][1];
        } else if ev.abs() < 1e-5 {
            leaf_vals[i][0] = 1.0 - leaves_vec[i][0];
            leaf_vals[i][1] = 1.0 - leaves_vec[i][1];
        } else {
            // -1: Marginalized out
            leaf_vals[i][0] = 1.0;
            leaf_vals[i][1] = 1.0;
        }
    }

    let p0 = leaf_vals[0][0] * leaf_vals[1][0] * leaf_vals[2][0] * leaf_vals[3][0];
    let p1 = leaf_vals[0][1] * leaf_vals[1][1] * leaf_vals[2][1] * leaf_vals[3][1];
    let spn_out = weights_vec[0] * p0 + weights_vec[1] * p1;

    Tensor::new(&[spn_out], leaves_p.device())
}

/// Evaluates bivariate Clayton copula CDF:
/// C(u, v) = max(u^(-theta) + v^(-theta) - 1, 0)^(-1/theta)
pub fn clayton_copula_cdf(u: &Tensor, v: &Tensor, theta: f64) -> Result<Tensor> {
    let u_vec: Vec<f32> = u.to_vec1()?;
    let v_vec: Vec<f32> = v.to_vec1()?;
    let n = u_vec.len();

    let mut c_vals = vec![0.0f32; n];
    let neg_inv_theta = -1.0 / theta;

    for i in 0..n {
        let u_val = u_vec[i] as f64;
        let v_val = v_vec[i] as f64;
        let inner = (u_val.powf(-theta) + v_val.powf(-theta) - 1.0).max(1e-7);
        c_vals[i] = inner.powf(neg_inv_theta) as f32;
    }

    Tensor::new(c_vals, u.device())
}

/// Evaluates Generalized Pareto Distribution (GPD) CDF for excesses y >= 0:
/// F(y) = 1 - (1 + xi * y / sigma)^(-1/xi)
pub fn gpd_tail_cdf(y: &Tensor, xi: f64, sigma: f64) -> Result<Tensor> {
    let y_vec: Vec<f32> = y.to_vec1()?;
    let n = y_vec.len();
    let mut out = vec![0.0f32; n];
    let neg_inv_xi = -1.0 / xi;

    for i in 0..n {
        let y_val = y_vec[i] as f64;
        let bracket = (1.0 + (xi * y_val) / sigma).max(1e-7);
        out[i] = (1.0 - bracket.powf(neg_inv_xi)) as f32;
    }

    Tensor::new(out, y.device())
}

/// Evaluates neural likelihood-to-marginal density ratio r(x, theta) = exp(logit).
pub fn sbi_neural_ratio(
    x: &Tensor,
    theta: &Tensor,
    w: &Tensor,
    b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let features = Tensor::cat(&[x, theta], 1)?;
    let logit = features.matmul(w)?.broadcast_add(b)?;
    let ratio = logit.exp()?;
    Ok((ratio, logit))
}
