//! CPU (and optional WGSL) kernels for Torch Export AOT Autograd and IR Inspection.

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

/// WGSL compute shader for decomposed Core ATen Softmax.
pub const ATEN_CORE_SOFTMAX_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_out: array<f32>;

struct Params {
    batch: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn softmax_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.batch) {
        return;
    }

    let offset = row * params.dim;

    // Step 1: Find max value in row
    var max_val = -3.402823e+38;
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let v = x_in[offset + i];
        if (v > max_val) {
            max_val = v;
        }
    }

    // Step 2: Compute exp and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < params.dim; i = i + 1u) {
        let e = exp(x_in[offset + i] - max_val);
        y_out[offset + i] = e;
        sum_exp = sum_exp + e;
    }

    // Step 3: Normalize
    for (var i = 0u; i < params.dim; i = i + 1u) {
        y_out[offset + i] = y_out[offset + i] / sum_exp;
    }
}
"#;

/// CPU reference implementation for Core ATen Softmax.
pub fn aten_core_softmax_cpu(x: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * dim];
    for b in 0..batch {
        let offset = b * dim;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..dim {
            let v = x[offset + i];
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0f32;
        for i in 0..dim {
            let e = (x[offset + i] - max_val).exp();
            y[offset + i] = e;
            sum_exp += e;
        }
        for i in 0..dim {
            y[offset + i] /= sum_exp;
        }
    }
    y
}
