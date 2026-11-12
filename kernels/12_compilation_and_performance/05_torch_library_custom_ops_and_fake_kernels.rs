//! CPU (and optional WGSL) kernels for Torch Library Custom Ops and Fake Kernels.

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

/// WGSL compute shader for custom L1 proximal soft-shrinkage operator.
pub const CUSTOM_SOFT_SHRINK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_out: array<f32>;

struct Params {
    total_elements: u32,
    lambd: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn soft_shrink_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let val = x_in[idx];
    var sign = 0.0;
    if (val > 0.0) {
        sign = 1.0;
    } else if (val < 0.0) {
        sign = -1.0;
    }

    let mag = max(abs(val) - params.lambd, 0.0);
    y_out[idx] = sign * mag;
}
"#;

/// CPU reference implementation for custom L1 proximal soft-shrinkage.
pub fn custom_soft_shrink_cpu(x: &[f32], lambd: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let sign = if v > 0.0 {
                1.0f32
            } else if v < 0.0 {
                -1.0f32
            } else {
                0.0f32
            };
            let mag = (v.abs() - lambd).max(0.0f32);
            sign * mag
        })
        .collect()
}
