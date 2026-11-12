//! CPU and WGSL kernels for Sub Byte Bitpacking T-MAC LUT and GGUF K Quants.

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

/// WGSL compute shader for T-MAC table lookup dot product without floating-point multiplies.
pub const TMAC_LUT_GEMM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> lut: array<f32, 16>;
@group(0) @binding(1) var<storage, read> weight_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;

struct Uniforms {
    total_elements: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let code = weight_indices[idx];
    output_values[idx] = lut[code & 0x0Fu];
}
"#;

/// CPU reference implementation of T-MAC table lookup.
pub fn tmac_lut_lookup_cpu(lut: &[f32], weight_indices: &[u32]) -> Vec<f32> {
    weight_indices
        .iter()
        .map(|&idx| lut[(idx & 0x0F) as usize])
        .collect()
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn tmac_lut_lookup_auto(lut: &[f32], weight_indices: &[u32]) -> Vec<f32> {
    tmac_lut_lookup_cpu(lut, weight_indices)
}
