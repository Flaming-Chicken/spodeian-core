//! Optimisation as Architecture LISTA ADMM-Net CRATE and Mesa-Optimizers
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Elementwise soft-thresholding: sign(x) * max(|x| - theta, 0)
pub fn soft_threshold(x: &Tensor, theta: &Tensor) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let sub = abs_x.broadcast_sub(theta)?;
    let clamped = sub.clamp(0.0f32, 1e9f32)?;
    let sign_x = x.sign()?;
    sign_x.mul(&clamped)
}

/// Learned ISTA (LISTA) layer unrolling:
/// z_{k+1} = soft_threshold( z_k W_s + y W_e, theta )
pub fn lista_unroll(
    y: &Tensor,
    w_e: &Tensor,
    w_s: &Tensor,
    theta: &Tensor,
    num_layers: usize,
) -> Result<Tensor> {
    let (b_sz, _) = y.dims2()?;
    let (_, n) = w_e.dims2()?;
    let mut z = Tensor::zeros((b_sz, n), y.dtype(), y.device())?;
    let y_we = y.matmul(w_e)?;

    for _ in 0..num_layers {
        let z_ws = z.matmul(w_s)?;
        let lin = z_ws.add(&y_we)?;
        z = soft_threshold(&lin, theta)?;
    }
    Ok(z)
}

/// ADMM-Net iteration step:
/// 1. x_{k+1} = (y + rho (z - u)) @ W_inv
/// 2. z_{k+1} = soft_threshold( x_{k+1} + u, theta / rho )
/// 3. u_{k+1} = u + x_{k+1} - z_{k+1}
pub fn admm_net_step(
    y: &Tensor,
    _x: &Tensor,
    z: &Tensor,
    u: &Tensor,
    w_inv: &Tensor,
    rho: f64,
    theta: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let z_sub_u = z.sub(u)?;
    let scaled = z_sub_u.affine(rho, 0.0)?;
    let rhs = y.add(&scaled)?;
    let x_next = rhs.matmul(w_inv)?;

    let x_add_u = x_next.add(u)?;
    let thresh_val = (theta / rho) as f32;
    let thresh_tensor = Tensor::from_slice(&[thresh_val], (1,), y.device())?;
    let z_next = soft_threshold(&x_add_u, &thresh_tensor)?;

    let diff = x_next.sub(&z_next)?;
    let u_next = u.add(&diff)?;

    Ok((x_next, z_next, u_next))
}

/// CRATE Step:
/// z_{k+1} = soft_threshold( z - eta * z @ (D^T D - I), lambda_reg )
pub fn crate_step(z: &Tensor, d_dict: &Tensor, step_size: f64, lambda_reg: f64) -> Result<Tensor> {
    let dt = d_dict.t()?;
    let dtd = dt.matmul(d_dict)?;
    let (_, n) = dtd.dims2()?;
    let eye = Tensor::eye(n, candle_core::DType::F32, z.device())?;
    let gram_eye = dtd.sub(&eye)?;

    let grad = z.matmul(&gram_eye)?;
    let step = grad.affine(step_size, 0.0)?;
    let z_step = z.sub(&step)?;

    let thresh_tensor = Tensor::from_slice(&[lambda_reg as f32], (1,), z.device())?;
    soft_threshold(&z_step, &thresh_tensor)
}
