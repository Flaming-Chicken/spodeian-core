//! CPU and WGSL kernels for QAT STE LSQ LSQ+ PACT and DoReFa.

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

/// WGSL compute shader for fused PACT activation clipping and quantization:
/// `x_q = round(clamp(x, 0.0, alpha) / (alpha / qmax)) * (alpha / qmax)`
pub const PACT_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_out: array<f32>;

struct Uniforms {
    total_elements: u32,
    alpha: f32,
    qmax: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let val = x_in[idx];
    let clipped = clamp(val, 0.0, params.alpha);
    let step = params.alpha / params.qmax;
    let q = round(clipped / step);
    x_out[idx] = q * step;
}
"#;

/// CPU reference implementation of fused PACT activation quantization.
pub fn pact_dequant_cpu(x_in: &[f32], alpha: f32, qmax: f32) -> Vec<f32> {
    let mut x_out = vec![0.0f32; x_in.len()];
    let step = alpha / qmax;
    for i in 0..x_in.len() {
        let clipped = x_in[i].clamp(0.0, alpha);
        let q = (clipped / step).round();
        x_out[i] = q * step;
    }
    x_out
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn pact_dequant_auto(x_in: &[f32], alpha: f32, qmax: f32) -> Vec<f32> {
    pact_dequant_cpu(x_in, alpha, qmax)
}
