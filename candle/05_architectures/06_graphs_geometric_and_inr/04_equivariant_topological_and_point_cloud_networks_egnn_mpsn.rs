//! Equivariant, Topological, and Point Cloud Networks: EGNN, MPSN, and PointNet
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// E(n)-Equivariant Graph Neural Network (EGNN) message passing layer.
pub fn egnn_layer(
    x: &Tensor,
    h: &Tensor,
    adj: &Tensor,
    w_m: &Tensor,
    b_m: &Tensor,
    w_x: &Tensor,
    b_x: &Tensor,
    w_h: &Tensor,
    b_h: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let n = h.dim(0)?;
    let d = h.dim(1)?;
    let d_m = b_m.dim(0)?;

    let x_i = x.unsqueeze(1)?;
    let x_j = x.unsqueeze(0)?;
    let delta_x = x_i.broadcast_sub(&x_j)?;
    let d_sq = delta_x.sqr()?.sum_keepdim(2)?;

    let h_i = h.unsqueeze(1)?.repeat((1, n, 1))?;
    let h_j = h.unsqueeze(0)?.repeat((n, 1, 1))?;
    let m_in = Tensor::cat(&[&h_i, &h_j, &d_sq], 2)?;

    let m_dim_in = 2 * d + 1;
    let m_in_flat = m_in.reshape((n * n, m_dim_in))?;
    let m_proj = m_in_flat.matmul(w_m)?.broadcast_add(b_m)?;
    let m_act = candle_nn::ops::silu(&m_proj)?;

    let adj_expanded = adj.unsqueeze(2)?;
    let m_ij = m_act.reshape((n, n, d_m))?.broadcast_mul(&adj_expanded)?;

    let m_ij_flat = m_ij.reshape((n * n, d_m))?;
    let coord_w_flat = m_ij_flat.matmul(w_x)?.broadcast_add(b_x)?;
    let coord_weights = coord_w_flat.reshape((n, n, 1))?;

    let delta_scaled = delta_x.broadcast_mul(&coord_weights)?;
    let x_update = delta_scaled.sum(1)?;
    let x_next = (x + x_update)?;

    let m_i = m_ij.sum(1)?;
    let h_in = Tensor::cat(&[h, &m_i], 1)?;
    let h_proj = h_in.matmul(w_h)?.broadcast_add(b_h)?;
    let h_act = candle_nn::ops::silu(&h_proj)?;
    let h_next = (h + h_act)?;

    Ok((x_next, h_next))
}

/// PointNet symmetric permutation-invariant global pooling: max_{p in P} ReLU(p @ W + b).
pub fn pointnet_global_pool(x: &Tensor, w_p: &Tensor, b_p: &Tensor) -> Result<Tensor> {
    let f_pre = x.matmul(w_p)?.broadcast_add(b_p)?;
    let f = f_pre.relu()?;
    f.max_keepdim(0)
}
