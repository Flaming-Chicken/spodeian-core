//! CPU and WGSL kernels for AWQ Activation Aware Weight Quantization.

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

/// WGSL shader for fused AWQ per-channel inverse scaling and dequantization:
/// W_eff[i, j] = (W_q[i, j] * col_scale[j]) / channel_scale[i]
pub const AWQ_SCALE_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> w_q: array<f32>;
@group(0) @binding(1) var<storage, read> channel_scale: array<f32>;
@group(0) @binding(2) var<storage, read> col_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_eff: array<f32>;

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
    let deq = w_q[idx] * col_scale[col];
    w_eff[idx] = deq / channel_scale[row];
}
"#;

/// CPU reference implementation of fused AWQ dequantization and inverse scaling.
pub fn awq_scale_dequant_cpu(
    w_q: &[f32],
    channel_scale: &[f32],
    col_scale: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut w_eff = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        let s_i = channel_scale[i];
        for j in 0..d_out {
            let deq = w_q[i * d_out + j] * col_scale[j];
            w_eff[i * d_out + j] = deq / s_i;
        }
    }
    w_eff
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn awq_scale_dequant_auto(
    w_q: &[f32],
    channel_scale: &[f32],
    col_scale: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    awq_scale_dequant_cpu(w_q, channel_scale, col_scale, d_in, d_out)
}
