//! Heterophilic, Temporal, Cellular Sheaves, and Knowledge Graph Embeddings
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Decoupled high-pass / low-pass spatial filtering for heterophilic graphs.
pub fn heterophilic_filter(
    h: &Tensor,
    norm_adj: &Tensor,
    w_self: &Tensor,
    w_low: &Tensor,
    w_high: &Tensor,
) -> Result<Tensor> {
    let n = norm_adj.dim(0)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, norm_adj.device())?;
    let high_pass = (eye - norm_adj)?;

    let h_self = h.matmul(w_self)?;
    let h_low = norm_adj.matmul(h)?.matmul(w_low)?;
    let h_high = high_pass.matmul(h)?.matmul(w_high)?;

    let sum_h = ((h_self + h_low)? + h_high)?;
    sum_h.relu()
}

/// Constructs 1D orthogonal cellular sheaf Laplacian: L_F = D - S * A.
pub fn cellular_sheaf_laplacian_1d(adj: &Tensor, edge_signs: &Tensor) -> Result<Tensor> {
    let n = adj.dim(0)?;
    let deg = adj.sum_keepdim(1)?.to_vec2::<f32>()?;

    let mut d_vec = vec![0.0f32; n * n];
    for i in 0..n {
        d_vec[i * n + i] = deg[i][0];
    }
    let d_mat = Tensor::from_vec(d_vec, (n, n), adj.device())?;
    let signed_adj = (edge_signs * adj)?;
    d_mat - signed_adj
}

/// Continuous cellular sheaf diffusion step: H_next = H - dt * L_F @ H.
pub fn sheaf_diffusion_step(h: &Tensor, sheaf_laplacian: &Tensor, dt: f64) -> Result<Tensor> {
    let diff = sheaf_laplacian.matmul(h)?.affine(dt, 0.0)?;
    h - diff
}

/// RotatE Knowledge Graph Embedding score in complex Euclidean space.
pub fn rotate_kge_score(
    h_re: &Tensor,
    h_im: &Tensor,
    theta: &Tensor,
    t_re: &Tensor,
    t_im: &Tensor,
) -> Result<Tensor> {
    let r_re = theta.cos()?;
    let r_im = theta.sin()?;

    let term1 = (h_re * &r_re)?;
    let term2 = (h_im * &r_im)?;
    let rot_re = (term1 - term2)?;

    let term3 = (h_re * &r_im)?;
    let term4 = (h_im * &r_re)?;
    let rot_im = (term3 + term4)?;

    let diff_re = (rot_re - t_re)?;
    let diff_im = (rot_im - t_im)?;

    let dist_sq = (diff_re.sqr()? + diff_im.sqr()?)?.sum_keepdim(1)?;
    dist_sq.affine(1.0, 1e-12)?.sqrt()?.affine(-1.0, 0.0)
}
