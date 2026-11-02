//! CPU and WGSL kernels for OmniQuant and HQQ Optimization-Based PTQ.

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

/// WGSL compute shader for fused LWC dequantization: W_deq[i, j] = q[i, j] * scale[j] + lower[j]
pub const LWC_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> q_in: array<f32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> lowers: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_deq: array<f32>;

struct Uniforms {
    d_in: u32,
    d_out: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= params.d_in || col >= params.d_out) {
        return;
    }
    let idx = row * params.d_out + col;
    w_deq[idx] = q_in[idx] * scales[col] + lowers[col];
}
"#;

/// CPU reference implementation of fused LWC asymmetric dequantization.
pub fn lwc_dequant_cpu(
    q_in: &[f32],
    scales: &[f32],
    lowers: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut w_deq = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            let idx = i * d_out + j;
            w_deq[idx] = q_in[idx] * scales[j] + lowers[j];
        }
    }
    w_deq
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn lwc_dequant_auto(
    q_in: &[f32],
    scales: &[f32],
    lowers: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    lwc_dequant_cpu(q_in, scales, lowers, d_in, d_out)
}
