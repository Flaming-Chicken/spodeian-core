//! CPU (and optional WGSL) kernels for Population Based Training PBT and DARTS Neural Architecture Search.

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

/// WGSL compute shader for DARTS mixed operator forward reduction across candidate operations.
pub const DARTS_MIXED_OP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> cand_ops: array<f32>;
@group(0) @binding(2) var<storage, read_write> mixed_out: array<f32>;

struct Params {
    num_edges: u32,
    num_ops: u32,
    spatial_size: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_elements = params.num_edges * params.spatial_size;
    let idx = gid.x;
    if (idx >= total_elements) {
        return;
    }
    let e = idx / params.spatial_size;
    let s = idx % params.spatial_size;

    var sum_val: f32 = 0.0;
    for (var m: u32 = 0u; m < params.num_ops; m = m + 1u) {
        let w = weights[e * params.num_ops + m];
        let op_val = cand_ops[(e * params.num_ops + m) * params.spatial_size + s];
        sum_val = sum_val + w * op_val;
    }
    mixed_out[idx] = sum_val;
}
"#;

/// CPU reference implementation of DARTS mixed operator forward reduction.
pub fn darts_mixed_op_cpu(
    weights: &[f32],
    cand_ops: &[f32],
    num_edges: usize,
    num_ops: usize,
    spatial_size: usize,
) -> Vec<f32> {
    let total_elements = num_edges * spatial_size;
    let mut out = vec![0.0f32; total_elements];
    for idx in 0..total_elements {
        let e = idx / spatial_size;
        let s = idx % spatial_size;
        let mut sum_val = 0.0f32;
        for m in 0..num_ops {
            let w = weights[e * num_ops + m];
            let op_val = cand_ops[(e * num_ops + m) * spatial_size + s];
            sum_val += w * op_val;
        }
        out[idx] = sum_val;
    }
    out
}

/// Auto-dispatching DARTS mixed operator forward reduction (CPU fallback).
pub fn darts_mixed_op_auto(
    weights: &[f32],
    cand_ops: &[f32],
    num_edges: usize,
    num_ops: usize,
    spatial_size: usize,
) -> Vec<f32> {
    darts_mixed_op_cpu(weights, cand_ops, num_edges, num_ops, spatial_size)
}
