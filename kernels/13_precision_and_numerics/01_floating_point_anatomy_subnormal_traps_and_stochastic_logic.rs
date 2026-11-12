//! CPU (and optional WGSL) kernels for Floating Point Anatomy Subnormal Traps and Stochastic Logic.

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

/// WGSL compute shader for hardware Flush-To-Zero (FTZ) simulation.
pub const FTZ_DAZ_SIMULATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_out: array<f32>;

struct Params {
    total_elements: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn ftz_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let val = x_in[idx];
    if (abs(val) < params.threshold) {
        y_out[idx] = 0.0;
    } else {
        y_out[idx] = val;
    }
}
"#;

/// CPU reference implementation for Flush-To-Zero simulation.
pub fn ftz_daz_simulate_cpu(x: &[f32], threshold: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| if v.abs() < threshold { 0.0f32 } else { v })
        .collect()
}
