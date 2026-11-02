//! CPU and WGSL kernels for 2:4 Semi-Structured Sparsity and RigL Dynamic Sparse.

/// WGSL compute shader for compressed 2:4 sparse matrix-vector multiplication.
/// Simulates NVIDIA Ampere Sparse Tensor Core 2:4 execution.
pub const SEMI_STRUCTURED_2_4_GEMM_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
    num_blocks: u32, // d_out / 4
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> packed_values: array<f32>; // [d_in, num_blocks * 2]
@group(0) @binding(3) var<storage, read> metadata_indices: array<u32>; // [d_in, num_blocks * 2]
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>; // [batch, d_out]

@compute @workgroup_size(16, 16, 1)
fn sparse_2_4_gemm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    let blk = gid.y;
    if (b >= u.batch || blk >= u.num_blocks) { return; }

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    for (var i = 0u; i < u.d_in; i++) {
        let x_val = x[b * u.d_in + i];
        let p_idx = i * (u.num_blocks * 2u) + blk * 2u;

        let v0 = packed_values[p_idx];
        let off0 = metadata_indices[p_idx];
        let v1 = packed_values[p_idx + 1u];
        let off1 = metadata_indices[p_idx + 1u];

        if (off0 == 0u) { sum0 += x_val * v0; }
        else if (off0 == 1u) { sum1 += x_val * v0; }
        else if (off0 == 2u) { sum2 += x_val * v0; }
        else if (off0 == 3u) { sum3 += x_val * v0; }

        if (off1 == 0u) { sum0 += x_val * v1; }
        else if (off1 == 1u) { sum1 += x_val * v1; }
        else if (off1 == 2u) { sum2 += x_val * v1; }
        else if (off1 == 3u) { sum3 += x_val * v1; }
    }

    let out_col = blk * 4u;
    y_out[b * u.d_out + out_col] = sum0;
    y_out[b * u.d_out + out_col + 1u] = sum1;
    y_out[b * u.d_out + out_col + 2u] = sum2;
    y_out[b * u.d_out + out_col + 3u] = sum3;
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

/// CPU 2:4 semi-structured pruning.
pub fn prune_2_to_4_cpu(w: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(cols % 4, 0);
    let mut w_sparse = vec![0.0f32; rows * cols];
    let mut mask = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..(cols / 4) {
            let offset = r * cols + b * 4;
            let block = [
                w[offset].abs(),
                w[offset + 1].abs(),
                w[offset + 2].abs(),
                w[offset + 3].abs(),
            ];
            let mut pairs = [(0, block[0]), (1, block[1]), (2, block[2]), (3, block[3])];
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            mask[offset + pairs[0].0] = 1.0;
            mask[offset + pairs[1].0] = 1.0;
            w_sparse[offset + pairs[0].0] = w[offset + pairs[0].0];
            w_sparse[offset + pairs[1].0] = w[offset + pairs[1].0];
        }
    }
    (w_sparse, mask)
}
