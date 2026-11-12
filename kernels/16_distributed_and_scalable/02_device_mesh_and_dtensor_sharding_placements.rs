//! CPU and WGSL kernels for Device Mesh and DTensor Sharding Placements.

/// WGSL compute shader for fused row-parallel partial shard AllReduce reduction.
pub const FUSED_DTENSOR_ROW_ALLREDUCE_WGSL: &str = r#"
struct Uniforms {
    mesh_size: u32,
    batch: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> partial_shards: array<f32>; // [mesh_size, batch, n]
@group(0) @binding(2) var<storage, read_write> allreduced_out: array<f32>; // [batch, n]

@compute @workgroup_size(16, 16, 1)
fn row_allreduce_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= u.batch || col >= u.n) { return; }

    var acc: f32 = 0.0;
    let slice_size = u.batch * u.n;
    for (var r = 0u; r < u.mesh_size; r++) {
        acc += partial_shards[r * slice_size + row * u.n + col];
    }

    allreduced_out[row * u.n + col] = acc;
}
"#;

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

/// CPU DTensor row-parallel Partial(Sum) AllReduce reduction: `out[b, c] = sum_r partial_shards[r, b, c]`.
pub fn dtensor_row_allreduce_cpu(
    partial_shards: &[f32],
    mesh_size: usize,
    batch: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * n];
    let slice_size = batch * n;

    for b in 0..batch {
        for col in 0..n {
            let mut acc = 0.0f32;
            for r in 0..mesh_size {
                acc += partial_shards[r * slice_size + b * n + col];
            }
            out[b * n + col] = acc;
        }
    }

    out
}
