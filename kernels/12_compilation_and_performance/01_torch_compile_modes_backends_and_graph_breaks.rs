//! CPU (and optional WGSL) kernels for Torch Compile Modes Backends and Graph Breaks.

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

/// WGSL compute shader for fused SwiGLU activation:
/// y = (x * sigmoid(x)) * gate.
pub const FUSED_SWIGLU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> gate_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out: array<f32>;

struct Params {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let x = x_in[idx];
    let sig = 1.0 / (1.0 + exp(-x));
    let silu = x * sig;
    y_out[idx] = silu * gate_in[idx];
}
"#;

/// CPU reference implementation of fused SwiGLU.
pub fn fused_swiglu_cpu(x: &[f32], gate: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(gate.iter())
        .map(|(&xv, &gv)| {
            let sig = 1.0f32 / (1.0f32 + (-xv).exp());
            (xv * sig) * gv
        })
        .collect()
}
