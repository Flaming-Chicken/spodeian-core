//! CPU and WGSL kernels for TorchAO Modern API and Affine Quantized Tensor (AQT).

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

/// WGSL shader for block-wise Affine Quantized Tensor (AQT) dequantization.
pub const AQT_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> q_int: array<f32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> dequant: array<f32>;

struct Uniforms {
    total_elements: u32,
    group_size: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let group_idx = idx / params.group_size;
    let s = scales[group_idx];
    dequant[idx] = q_int[idx] * s;
}
"#;

/// CPU reference implementation of block-wise AQT dequantization.
pub fn aqt_dequant_cpu(q_int: &[f32], scales: &[f32], group_size: usize) -> Vec<f32> {
    let mut dequant = Vec::with_capacity(q_int.len());
    for (g, chunk) in q_int.chunks(group_size).enumerate() {
        let s = scales[g];
        for &v in chunk {
            dequant.push(v * s);
        }
    }
    dequant
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn aqt_dequant_auto(q_int: &[f32], scales: &[f32], group_size: usize) -> Vec<f32> {
    aqt_dequant_cpu(q_int, scales, group_size)
}
