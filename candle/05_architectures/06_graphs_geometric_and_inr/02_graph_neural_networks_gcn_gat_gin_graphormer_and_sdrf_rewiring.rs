//! Graph Neural Networks GCN, GAT, GIN, Graphormer, and SDRF Rewiring
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Symmetric normalized adjacency matrix with added self-loops: D^{-1/2} (A + I) D^{-1/2}.
pub fn gcn_normalize_adjacency(adj: &Tensor) -> Result<Tensor> {
    let n = adj.dim(0)?;
    let eye = Tensor::eye(n, candle_core::DType::F32, adj.device())?;
    let a_tilde = (adj + eye)?;
    let deg = a_tilde.sum_keepdim(1)?;
    let deg_vec = deg.to_vec2::<f32>()?;

    let mut d_vec = vec![0.0f32; n * n];
    for i in 0..n {
        let val = deg_vec[i][0];
        let inv_sqrt = if val > 0.0 { 1.0 / val.sqrt() } else { 0.0 };
        d_vec[i * n + i] = inv_sqrt;
    }
    let d_mat = Tensor::from_vec(d_vec, (n, n), adj.device())?;
    d_mat.matmul(&a_tilde)?.matmul(&d_mat)
}

/// GCN message passing layer: H_next = ReLU(A_hat @ H @ W + b).
pub fn gcn_layer(h: &Tensor, norm_adj: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let h_proj = h.matmul(weight)?;
    let h_conv = norm_adj.matmul(&h_proj)?.broadcast_add(bias)?;
    h_conv.relu()
}

/// GIN message passing layer: MLP((1 + eps) * h + A @ h).
pub fn gin_layer(
    h: &Tensor,
    adj: &Tensor,
    eps: f64,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let h_agg = adj.matmul(h)?;
    let h_scaled = h.affine(1.0 + eps, 0.0)?;
    let z = (h_scaled + h_agg)?;
    let h1_pre = z.matmul(w1)?.broadcast_add(b1)?;
    let h1 = h1_pre.relu()?;
    h1.matmul(w2)?.broadcast_add(b2)
}

/// Graphormer all-pairs attention with shortest path spatial bias.
pub fn graphormer_attention(
    h: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    spatial_bias: &Tensor,
) -> Result<Tensor> {
    let d = h.dim(1)?;
    let scale = 1.0 / (d as f64).sqrt();
    let q = h.matmul(w_q)?;
    let k = h.matmul(w_k)?;
    let v = h.matmul(w_v)?;

    let scores = (q.matmul(&k.t()?)?.affine(scale, 0.0)? + spatial_bias)?;
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    attn.matmul(&v)
}

/// Balanced Forman-Ricci curvature proxy for SDRF oversquashing analysis.
pub fn sdrf_ricci_curvature(adj: &Tensor) -> Result<Tensor> {
    let deg = adj.sum_keepdim(1)?;
    let deg_t = deg.t()?;
    let deg_sum = deg.broadcast_add(&deg_t)?;
    let triangles = adj.matmul(adj)?;
    let three_triangles = triangles.affine(3.0, 0.0)?;
    let base = deg_sum.affine(-1.0, 4.0)?;
    let curv = (base + three_triangles)?;
    curv * adj
}
