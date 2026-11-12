//! Numerical Integration PCEC RK45 Neural ODE LTC and SDE
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Evaluates neural ODE vector field f(x) = tanh(x W1 + b1) W2 + b2.
pub fn eval_vector_field(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    x.matmul(w1)?
        .broadcast_add(b1)?
        .tanh()?
        .matmul(w2)?
        .broadcast_add(b2)
}

/// Computes analytical Jacobian matrix J_{ij} = df_j / dx_i of shape [B, D_in, D_out].
pub fn eval_jacobian(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor) -> Result<Tensor> {
    let b = x.dim(0)?;
    let z = x.matmul(w1)?.broadcast_add(b1)?;
    let a = z.tanh()?;
    let one = Tensor::ones_like(&a)?;
    let d_a = (one - a.sqr()?)?;

    let mut j_rows = Vec::with_capacity(b);
    for i in 0..b {
        let da_i = d_a.narrow(0, i, 1)?; // [1, 4]
        let w1_scaled = w1.broadcast_mul(&da_i)?; // [2, 4]
        let j_i = w1_scaled.matmul(w2)?; // [2, 2]
        j_rows.push(j_i.unsqueeze(0)?);
    }
    let j_refs: Vec<&Tensor> = j_rows.iter().collect();
    Tensor::cat(&j_refs, 0)
}

/// Evaluates 4 numerical ODE integrators: Euler, Heun (RK2), PCEC, and RK4.
pub fn integrators_step(
    x0: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    h: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let f = |x: &Tensor| eval_vector_field(x, w1, b1, w2, b2);
    let f0 = f(x0)?;

    // 1. Explicit Euler: x0 + h * f0
    let x_euler = (x0 + f0.affine(h, 0.0)?)?;

    // 2. Heun's 2nd-order Runge-Kutta:
    let k1 = f0.clone();
    let k2 = f(&(x0 + k1.affine(h, 0.0)?)?)?;
    let x_heun = (x0 + (k1.clone() + k2)?.affine(0.5 * h, 0.0)?)?;

    // 3. Predictor-Corrector PCEC:
    let x_p = (x0 + f0.affine(h, 0.0)?)?;
    let f_p = f(&x_p)?;
    let x_c1 = (x0 + (&f0 + f_p)?.affine(0.5 * h, 0.0)?)?;
    let f_c = f(&x_c1)?;
    let x_pcec = (x0 + (&f0 + f_c)?.affine(0.5 * h, 0.0)?)?;

    // 4. Runge-Kutta 4th-order (RK4):
    let rk_k1 = f0.clone();
    let rk_k2 = f(&(x0 + rk_k1.affine(0.5 * h, 0.0)?)?)?;
    let rk_k3 = f(&(x0 + rk_k2.affine(0.5 * h, 0.0)?)?)?;
    let rk_k4 = f(&(x0 + rk_k3.affine(h, 0.0)?)?)?;

    let sum_k = (rk_k1 + rk_k2.affine(2.0, 0.0)? + rk_k3.affine(2.0, 0.0)? + rk_k4)?;
    let x_rk4 = (x0 + sum_k.affine(h / 6.0, 0.0)?)?;

    Ok((f0, x_euler, x_heun, x_pcec, x_rk4))
}

/// Continuous Adjoint sensitivity backward step: a(0) = a(T) + h * (a(T) @ J_f).
pub fn adjoint_step(adj_t: &Tensor, j0: &Tensor, h: f64) -> Result<Tensor> {
    let b = adj_t.dim(0)?;
    let mut a0_rows = Vec::with_capacity(b);

    for i in 0..b {
        let a_i = adj_t.narrow(0, i, 1)?; // [1, 2]
        let j_i = j0.narrow(0, i, 1)?.squeeze(0)?; // [2, 2]
        let da = a_i.matmul(&j_i.t()?)?; // [1, 2]
        let a0_i = (a_i + da.affine(h, 0.0)?)?;
        a0_rows.push(a0_i);
    }
    let a0_refs: Vec<&Tensor> = a0_rows.iter().collect();
    Tensor::cat(&a0_refs, 0)
}

/// Closed-Form Continuous-Time (CfC) network evaluation.
pub fn cfc_step(
    x0: &Tensor,
    f0: &Tensor,
    w_cfc1: &Tensor,
    w_cfc2: &Tensor,
    h: f64,
) -> Result<Tensor> {
    let neg_abs_f_h = f0.abs()?.neg()?.affine(h, 0.0)?;
    let gate = candle_nn::ops::sigmoid(&neg_abs_f_h)?;

    let one = Tensor::ones_like(&gate)?;
    let one_minus_gate = (one - &gate)?;

    let term1 = x0.matmul(w_cfc1)?.mul(&gate)?;
    let term2 = x0.matmul(w_cfc2)?.mul(&one_minus_gate)?;
    term1 + term2
}

/// Neural SDE integration: Euler-Maruyama vs. Milstein.
pub fn sde_step(
    x0: &Tensor,
    f0: &Tensor,
    dw: &Tensor,
    sigma: f64,
    h: f64,
) -> Result<(Tensor, Tensor)> {
    let x_em = (x0 + f0.affine(h, 0.0)? + x0.mul(dw)?.affine(sigma, 0.0)?)?;

    let dw_sq_minus_h = (dw.sqr()? - h)?;
    let correction = x0.affine(0.5 * sigma * sigma, 0.0)?.mul(&dw_sq_minus_h)?;
    let x_milstein = (x_em.clone() + correction)?;

    Ok((x_em, x_milstein))
}
