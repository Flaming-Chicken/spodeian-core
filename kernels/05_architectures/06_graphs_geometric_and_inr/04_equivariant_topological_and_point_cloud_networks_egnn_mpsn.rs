//! CPU (and optional WGSL) kernels for Equivariant, Topological, and Point Cloud Networks: EGNN, MPSN, and PointNet.

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

/// Computes pairwise squared Euclidean distances d_{ij}^2 on CPU: invariant under E(n).
pub fn pairwise_distances_cpu(coords: &[f32], num_points: usize) -> Vec<f32> {
    let mut d_sq = vec![0.0f32; num_points * num_points];
    for i in 0..num_points {
        for j in 0..num_points {
            let dx = coords[i * 3 + 0] - coords[j * 3 + 0];
            let dy = coords[i * 3 + 1] - coords[j * 3 + 1];
            let dz = coords[i * 3 + 2] - coords[j * 3 + 2];
            d_sq[i * num_points + j] = dx * dx + dy * dy + dz * dz;
        }
    }
    d_sq
}

/// PointNet symmetric global max pooling on CPU.
pub fn pointnet_global_pool_cpu(features: &[f32], num_points: usize, dim: usize) -> Vec<f32> {
    let mut global_max = vec![f32::NEG_INFINITY; dim];
    for i in 0..num_points {
        for d in 0..dim {
            let val = features[i * dim + d].max(0.0); // ReLU
            if val > global_max[d] {
                global_max[d] = val;
            }
        }
    }
    global_max
}

/// WGSL shader for parallel pairwise squared distance computation in 3D.
pub const WGSL_PAIRWISE_DISTANCE: &str = r#"
@group(0) @binding(0) var<storage, read> coords: array<f32>;
@group(0) @binding(1) var<storage, read_write> d_sq: array<f32>;

struct Uniforms {
    num_points: u32,
};
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.num_points * u.num_points;
    if (idx >= total) {
        return;
    }
    let i = idx / u.num_points;
    let j = idx % u.num_points;

    let dx = coords[i * 3u + 0u] - coords[j * 3u + 0u];
    let dy = coords[i * 3u + 1u] - coords[j * 3u + 1u];
    let dz = coords[i * 3u + 2u] - coords[j * 3u + 2u];

    d_sq[idx] = dx * dx + dy * dy + dz * dz;
}
"#;
