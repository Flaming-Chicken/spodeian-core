//! Hypergraph Neural Networks: AllSet, Combinatorial Complexes, and Exphormer
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Hypergraph convolution via bipartite incidence matrix H in R^{V x E}.
pub fn hypergraph_conv(
    x: &Tensor,
    inc: &Tensor,
    w_feat: &Tensor,
    b_feat: &Tensor,
) -> Result<Tensor> {
    let d_v = inc.sum_keepdim(1)?;
    let d_e = inc.sum_keepdim(0)?;

    let z_e = inc.t()?.matmul(x)?.broadcast_div(&d_e.t()?)?;
    let x_agg = inc.matmul(&z_e)?.broadcast_div(&d_v)?;
    let proj = x_agg.matmul(w_feat)?.broadcast_add(b_feat)?;
    proj.gelu_erf()
}

/// AllSet universal multiset propagation step on hypergraphs.
pub fn allset_multiset_step(
    x: &Tensor,
    inc: &Tensor,
    w_v: &Tensor,
    b_v: &Tensor,
    w_e: &Tensor,
    b_e: &Tensor,
) -> Result<Tensor> {
    let h_v = (x.matmul(w_v)?.broadcast_add(b_v)?).relu()?;
    let z_e = (inc.t()?.matmul(&h_v)?.matmul(w_e)?.broadcast_add(b_e)?).relu()?;
    let x_rec = (inc.matmul(&z_e)?.matmul(w_v)?.broadcast_add(b_v)?).relu()?;
    x + x_rec
}

/// Exphormer expander graph sparse message mixing step.
pub fn exphormer_spectral_mixing(
    h: &Tensor,
    expander_adj: &Tensor,
    w_mix: &Tensor,
    b_mix: &Tensor,
) -> Result<Tensor> {
    let h_exp = expander_adj.matmul(h)?;
    let sum_h = (h + h_exp)?;
    let proj = sum_h.matmul(w_mix)?.broadcast_add(b_mix)?;
    proj.gelu_erf()
}
