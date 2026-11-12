//! CPU (and optional WGSL) kernels for Graph Neural Networks GCN, GAT, GIN, Graphormer, and SDRF Rewiring.

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

/// GCN neighborhood message aggregation on CPU: Out = A_hat @ H.
pub fn gcn_aggregate_cpu(
    norm_adj: &[f32],
    h_proj: &[f32],
    num_nodes: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_nodes * dim];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let a_ij = norm_adj[i * num_nodes + j];
            if a_ij != 0.0 {
                for d in 0..dim {
                    out[i * dim + d] += a_ij * h_proj[j * dim + d];
                }
            }
        }
    }
    out
}

/// Balanced Forman-Ricci curvature calculation for graph edges on CPU.
pub fn sdrf_ricci_curvature_cpu(adj: &[f32], num_nodes: usize) -> Vec<f32> {
    // 1. Calculate node degrees
    let mut deg = vec![0.0f32; num_nodes];
    for i in 0..num_nodes {
        deg[i] = (0..num_nodes).map(|j| adj[i * num_nodes + j]).sum();
    }

    // 2. Calculate triangles on edges: Tri(i, j) = sum_k adj[i, k] * adj[j, k]
    let mut ricci = vec![0.0f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let edge = adj[i * num_nodes + j];
            if edge > 0.0 {
                let mut tri = 0.0f32;
                for k in 0..num_nodes {
                    tri += adj[i * num_nodes + k] * adj[j * num_nodes + k];
                }
                ricci[i * num_nodes + j] = 4.0 - deg[i] - deg[j] + 3.0 * tri;
            }
        }
    }
    ricci
}

/// WGSL shader for parallel graph sparse/dense neighborhood aggregation.
pub const WGSL_GCN_AGGREGATE: &str = r#"
@group(0) @binding(0) var<storage, read> norm_adj: array<f32>;
@group(0) @binding(1) var<storage, read> h_proj: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Uniforms {
    num_nodes: u32,
    dim: u32,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_nodes * u.dim;
    if (idx >= total) {
        return;
    }
    let i = idx / u.dim;
    let d = idx % u.dim;

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < u.num_nodes; j = j + 1u) {
        let a_ij = norm_adj[i * u.num_nodes + j];
        if (a_ij != 0.0) {
            sum = sum + a_ij * h_proj[j * u.dim + d];
        }
    }
    out[idx] = sum;
}
"#;
