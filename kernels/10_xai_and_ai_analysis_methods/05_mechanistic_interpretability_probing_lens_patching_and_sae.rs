//! CPU (and optional WGSL) kernels for Mechanistic Interpretability Probing Lens Patching and SAE.

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

/// WGSL compute shader for JumpReLU hard-thresholding feature activation.
pub const JUMPRELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;

struct Params {
    num_elements: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }
    let val = inputs[idx];
    if (val > params.threshold) {
        outputs[idx] = val;
    } else {
        outputs[idx] = 0.0;
    }
}
"#;

/// CPU reference implementation of JumpReLU activation.
pub fn jumprelu_cpu(inputs: &[f32], threshold: f32) -> Vec<f32> {
    inputs
        .iter()
        .map(|&v| if v > threshold { v } else { 0.0f32 })
        .collect()
}
