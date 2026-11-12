//! Category 05 Topic 06: Lossless Deliberation: Reversible RevNet, Hyper-Connections, and Recurrent Looped Depth.
//!
//! Provides native Candle implementations for:
//! 1. Reversible RevNet coupling and exact algebraic inverse reconstruction.
//! 2. Multi-lane Hyper-Connections (HC).
//! 3. Looped Recurrent Depth with persistent input anchoring.

use candle_core::{Result, Tensor};

/// Evaluates a 2-layer MLP on a 3D tensor: [B, S, D_in] -> [B, S, D_out].
pub fn eval_mlp_3d(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let (b_sz, s_len, in_dim) = x.dims3()?;
    let out_dim = w2.dim(1)?;

    let x_flat = x.reshape((b_sz * s_len, in_dim))?;
    let h = x_flat.matmul(w1)?.broadcast_add(b1)?.gelu_erf()?;
    let out_flat = h.matmul(w2)?.broadcast_add(b2)?;
    out_flat.reshape((b_sz, s_len, out_dim))
}

/// Computes Layer Normalization along the trailing dimension.
pub fn layer_norm_last_dim(x: &Tensor, eps: f64) -> Result<Tensor> {
    let last_dim = x.rank() - 1;
    let mean = x.mean_keepdim(last_dim)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(last_dim)?;
    let std = (var + eps)?.sqrt()?;
    diff.broadcast_div(&std)
}

/// Reversible RevNet forward pass.
pub fn revnet_forward(
    x: &Tensor,
    f_w1: &Tensor,
    f_b1: &Tensor,
    f_w2: &Tensor,
    f_b2: &Tensor,
    g_w1: &Tensor,
    g_b1: &Tensor,
    g_w2: &Tensor,
    g_b2: &Tensor,
) -> Result<Tensor> {
    let d_sub = x.dim(2)? / 2;
    let x1 = x.narrow(2, 0, d_sub)?;
    let x2 = x.narrow(2, d_sub, d_sub)?;

    let f_x2 = eval_mlp_3d(&x2, f_w1, f_b1, f_w2, f_b2)?;
    let y1 = x1.add(&f_x2)?;

    let g_y1 = eval_mlp_3d(&y1, g_w1, g_b1, g_w2, g_b2)?;
    let y2 = x2.add(&g_y1)?;

    Tensor::cat(&[&y1, &y2], 2)
}

/// Reversible RevNet exact algebraic inverse reconstruction.
pub fn revnet_inverse(
    y: &Tensor,
    f_w1: &Tensor,
    f_b1: &Tensor,
    f_w2: &Tensor,
    f_b2: &Tensor,
    g_w1: &Tensor,
    g_b1: &Tensor,
    g_w2: &Tensor,
    g_b2: &Tensor,
) -> Result<Tensor> {
    let d_sub = y.dim(2)? / 2;
    let y1 = y.narrow(2, 0, d_sub)?;
    let y2 = y.narrow(2, d_sub, d_sub)?;

    let g_y1 = eval_mlp_3d(&y1, g_w1, g_b1, g_w2, g_b2)?;
    let x2 = y2.sub(&g_y1)?;

    let f_x2 = eval_mlp_3d(&x2, f_w1, f_b1, f_w2, f_b2)?;
    let x1 = y1.sub(&f_x2)?;

    Tensor::cat(&[&x1, &x2], 2)
}

/// Multi-lane Hyper-Connections forward pass.
pub fn hyper_connections_forward(
    h1: &Tensor,
    h2: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    hc_w1: &Tensor,
    hc_b1: &Tensor,
    hc_w2: &Tensor,
    hc_b2: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let a_vec = alpha.to_vec1::<f32>()?;
    let b_vec = beta.to_vec1::<f32>()?;

    let a0 = a_vec[0] as f64;
    let a1 = a_vec[1] as f64;
    let b0 = b_vec[0] as f64;
    let b1 = b_vec[1] as f64;

    let u = ((h1 * a0)? + (h2 * a1)?)?;
    let v = eval_mlp_3d(&u, hc_w1, hc_b1, hc_w2, hc_b2)?;

    let h1_next = (h1 + (&v * b0)?)?;
    let h2_next = (h2 + (&v * b1)?)?;
    let combined = ((&h1_next + &h2_next)? * 0.5)?;

    Ok((h1_next, h2_next, combined))
}

/// Looped Recurrent Depth forward pass with persistent input anchoring.
pub fn looped_depth_forward(
    x: &Tensor,
    loop_w1: &Tensor,
    loop_b1: &Tensor,
    loop_w2: &Tensor,
    loop_b2: &Tensor,
    steps: usize,
    gamma: f64,
) -> Result<Tensor> {
    let mut z = x.clone();
    for _ in 0..steps {
        let z_in = (&z + (x * gamma)?)?;
        let delta = eval_mlp_3d(&z_in, loop_w1, loop_b1, loop_w2, loop_b2)?;
        z = layer_norm_last_dim(&(&z + &delta)?, 1e-5)?;
    }
    Ok(z)
}
