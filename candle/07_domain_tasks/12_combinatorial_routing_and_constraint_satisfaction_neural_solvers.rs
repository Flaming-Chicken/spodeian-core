//! Combinatorial Routing and Constraint Satisfaction Neural Solvers
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes closed-loop Euclidean tour length for TSP.
pub fn tsp_tour_cost(coords: &Tensor, tour: &Tensor) -> Result<Tensor> {
    let (b, n, _) = coords.dims3()?;
    let coords_vec = coords.to_vec3::<f32>()?;
    let tour_vec = tour.to_vec2::<i64>()?;

    let mut costs = Vec::with_capacity(b);
    for bi in 0..b {
        let mut total_len = 0.0f32;
        let c = &coords_vec[bi];
        let t = &tour_vec[bi];
        for i in 0..n {
            let curr_idx = t[i] as usize;
            let next_idx = t[(i + 1) % n] as usize;
            let dx = c[curr_idx][0] - c[next_idx][0];
            let dy = c[curr_idx][1] - c[next_idx][1];
            total_len += (dx * dx + dy * dy + 1e-12f32).sqrt();
        }
        costs.push(total_len);
    }

    Tensor::new(costs, coords.device())
}

/// Pointer Network attention step to select next node.
pub fn pointer_network_step(
    node_embeds: &Tensor,
    decoder_state: &Tensor,
    visited_mask: &Tensor,
    w1: &Tensor,
    w2: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let (b, n, d) = node_embeds.dims3()?;
    let nodes_flat = node_embeds.reshape((b * n, d))?;
    let e1 = nodes_flat.matmul(w1)?.reshape((b, n, d))?;
    let e2 = decoder_state.matmul(w2)?.unsqueeze(1)?;
    let tanh_act = e1.broadcast_add(&e2)?.tanh()?;

    let act_flat = tanh_act.reshape((b * n, d))?;
    let scores = act_flat.matmul(&v.unsqueeze(1)?)?.reshape((b, n))?;

    let mask_val = (visited_mask * (-1e9))?;
    let one_minus_visited = (1.0 - visited_mask)?;
    let keep_val = (one_minus_visited * &scores)?;
    let masked_scores = (keep_val + mask_val)?;

    candle_nn::ops::softmax(&masked_scores, 1)
}

/// GNN Edge Classification / Tour Heatmap generation.
pub fn gnn_edge_heatmap(
    h_nodes: &Tensor,
    edge_dists: &Tensor,
    w_edge: &Tensor,
    b_edge: &Tensor,
) -> Result<Tensor> {
    let (b, n, d) = h_nodes.dims3()?;
    let hi = h_nodes.unsqueeze(2)?.broadcast_as((b, n, n, d))?;
    let hj = h_nodes.unsqueeze(1)?.broadcast_as((b, n, n, d))?;
    let dist_exp = edge_dists.unsqueeze(3)?;

    let edge_feats = Tensor::cat(&[&hi, &hj, &dist_exp], 3)?;
    let edge_flat = edge_feats.reshape((b * n * n, 2 * d + 1))?;
    let logits_flat = edge_flat.matmul(w_edge)?.broadcast_add(b_edge)?;
    let logits = logits_flat.reshape((b, n, n))?;

    candle_nn::ops::sigmoid(&logits)
}

/// Continuous relaxation of Max-Cut Hamiltonian.
pub fn relaxed_maxcut_cost(z_spins: &Tensor, w_adj: &Tensor) -> Result<Tensor> {
    let sigma = z_spins.tanh()?;
    let spin_outer = sigma.unsqueeze(2)?.broadcast_mul(&sigma.unsqueeze(1)?)?;
    let one_minus_outer = (1.0 - spin_outer)?;
    let cut_matrix = ((w_adj * 0.25)? * one_minus_outer)?;
    cut_matrix.sum(2)?.sum(1)
}
