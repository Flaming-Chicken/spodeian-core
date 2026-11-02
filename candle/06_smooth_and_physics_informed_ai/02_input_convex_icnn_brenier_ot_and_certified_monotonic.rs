//! Input Convex Neural Networks (ICNN), Brenier Optimal Transport, and Certified Monotonic Networks
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()?.affine(1.0, 1.0))?.log()
}

/// Evaluates scalar convex potential psi(x) via 2-layer ICNN.
pub fn icnn_potential(
    x: &Tensor,
    w_x0: &Tensor,
    b0: &Tensor,
    w_z1: &Tensor,
    w_x1: &Tensor,
    b1: &Tensor,
) -> Result<Tensor> {
    let w_z1_pos = softplus(w_z1)?;
    let z0 = x.matmul(w_x0)?.broadcast_add(b0)?.relu()?;
    let part_z = z0.matmul(&w_z1_pos)?;
    let part_x = x.matmul(w_x1)?;
    part_z.broadcast_add(&part_x)?.broadcast_add(b1)
}

/// Evaluates certified strictly monotone Brenier optimal transport map T(x) = x + nabla_x psi(x).
pub fn brenier_ot_gradient_map(
    x: &Tensor,
    w_x0: &Tensor,
    b0: &Tensor,
    w_z1: &Tensor,
    w_x1: &Tensor,
) -> Result<Tensor> {
    let w_z1_pos = softplus(w_z1)?;
    let pre_act = x.matmul(w_x0)?.broadcast_add(b0)?;
    let mask = pre_act.gt(0.0f32)?.to_dtype(DType::F32)?;
    let effective_z = mask.broadcast_mul(&w_z1_pos.t()?)?;
    let grad_psi = effective_z.matmul(&w_x0.t()?)?.broadcast_add(&w_x1.t()?)?;
    x.add(&grad_psi)
}
