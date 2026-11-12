//! CPU (and optional WGSL) kernels for Hypergraph Neural Networks: AllSet, Combinatorial Complexes, and Exphormer.

/// CPU linear anchor: `y[b, o] = bias[o] + sum_i x[b, i] * w[i, o]`.
pub fn linear_anchor_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for (b, row) in y.chunks_mut(d_out).enumerate() {
        for (o, out) in row.iter_mut().enumerate() {
            *out = bias[o]
                + (0..d_in)
                    .map(|i| x[b * d_in + i] * w[i * d_out + o])
                    .sum::<f32>();
        }
    }
    y
}

/// Node-to-Hyperedge incidence aggregation on CPU: Z_e = D_e^{-1} H^T X.
pub fn hypergraph_node_to_edge_cpu(
    x: &[f32],
    inc: &[f32],
    num_nodes: usize,
    num_edges: usize,
    dim: usize,
) -> Vec<f32> {
    // 1. Compute hyperedge degrees
    let mut d_e = vec![0.0f32; num_edges];
    for v in 0..num_nodes {
        for e in 0..num_edges {
            d_e[e] += inc[v * num_edges + e];
        }
    }

    // 2. Accumulate node features into hyperedge representations
    let mut z_e = vec![0.0f32; num_edges * dim];
    for e in 0..num_edges {
        let deg = d_e[e].max(1e-6);
        for v in 0..num_nodes {
            let h_ve = inc[v * num_edges + e];
            if h_ve > 0.0 {
                for d in 0..dim {
                    z_e[e * dim + d] += (h_ve * x[v * dim + d]) / deg;
                }
            }
        }
    }
    z_e
}

/// Exphormer expander graph message mixing on CPU.
pub fn exphormer_mixing_cpu(
    h: &[f32],
    expander_adj: &[f32],
    num_nodes: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = h.to_vec();
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let a_ij = expander_adj[i * num_nodes + j];
            if a_ij > 0.0 {
                for d in 0..dim {
                    out[i * dim + d] += a_ij * h[j * dim + d];
                }
            }
        }
    }
    out
}

/// WGSL compute shader for node-to-hyperedge projection.
pub const WGSL_HYPERGRAPH_INCIDENCE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> inc: array<f32>;
@group(0) @binding(2) var<storage, read_write> z_e: array<f32>;

struct Uniforms {
    num_nodes: u32,
    num_edges: u32,
    dim: u32,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_edges * u.dim;
    if (idx >= total) {
        return;
    }
    let e = idx / u.dim;
    let d = idx % u.dim;

    // Compute edge degree
    var deg: f32 = 0.0;
    for (var v: u32 = 0u; v < u.num_nodes; v = v + 1u) {
        deg = deg + inc[v * u.num_edges + e];
    }
    if (deg < 1e-6) {
        deg = 1.0;
    }

    var sum: f32 = 0.0;
    for (var v: u32 = 0u; v < u.num_nodes; v = v + 1u) {
        let h_ve = inc[v * u.num_edges + e];
        if (h_ve > 0.0) {
            sum = sum + h_ve * x[v * u.dim + d];
        }
    }
    z_e[idx] = sum / deg;
}
"#;
