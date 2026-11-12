//! Flow Matching Rectified Flow and Consistency Models
//! in candle/05_architectures/05_generative_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Conditional flow matching linear interpolation and constant target velocity.
pub fn conditional_flow_matching(
    x_0: &Tensor,
    x_1: &Tensor,
    t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let one_minus_t = t.affine(-1.0, 1.0)?;
    let term0 = x_0.broadcast_mul(&one_minus_t)?;
    let term1 = x_1.broadcast_mul(t)?;
    let x_t = (term0 + term1)?;
    let v_target = (x_1 - x_0)?;
    Ok((x_t, v_target))
}

/// Velocity prediction network v_theta(x, t).
pub fn velocity_network(
    x: &Tensor,
    t: &Tensor,
    w_x: &Tensor,
    b_x: &Tensor,
    w_t: &Tensor,
    b_t: &Tensor,
    w_out: &Tensor,
    b_out: &Tensor,
) -> Result<Tensor> {
    let t_proj = t.matmul(w_t)?.broadcast_add(b_t)?;
    let t_emb = candle_nn::ops::silu(&t_proj)?;
    let x_proj = x.matmul(w_x)?.broadcast_add(b_x)?;
    let h_pre = (x_proj + t_emb)?;
    let h = candle_nn::ops::silu(&h_pre)?;
    h.matmul(w_out)?.broadcast_add(b_out)
}

/// Single-step Euler ODE solver for rectified flow.
pub fn rectified_flow_step(x: &Tensor, v_pred: &Tensor, dt: f64) -> Result<Tensor> {
    let v_scaled = v_pred.affine(dt, 0.0)?;
    x + v_scaled
}

/// Consistency boundary condition coefficients: c_skip(t) and c_out(t).
pub fn consistency_coefficients(
    t: &Tensor,
    epsilon: f64,
    sigma_data: f64,
) -> Result<(Tensor, Tensor)> {
    let dt = t.affine(1.0, -epsilon)?;
    let sigma2 = sigma_data * sigma_data;
    let dt2 = dt.sqr()?;
    let c_skip = dt2.affine(1.0, sigma2)?.recip()?.affine(sigma2, 0.0)?;
    let t2 = t.sqr()?;
    let denom = t2.affine(1.0, sigma2)?.sqrt()?;
    let c_out = (dt.affine(sigma_data, 0.0)? / denom)?;
    Ok((c_skip, c_out))
}

/// Consistency model parameterization enforcing f_theta(x, epsilon) = x.
pub fn consistency_forward(
    x: &Tensor,
    t: &Tensor,
    f_x: &Tensor,
    epsilon: f64,
    sigma_data: f64,
) -> Result<Tensor> {
    let (c_skip, c_out) = consistency_coefficients(t, epsilon, sigma_data)?;
    let term0 = x.broadcast_mul(&c_skip)?;
    let term1 = f_x.broadcast_mul(&c_out)?;
    term0 + term1
}
