//! CPU and WGSL kernels for Calibration Observers (MinMax, Percentile, MSE, KL, ACIQ).

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

/// WGSL shader for calibrated symmetric quantization & clipping.
pub const CALIBRATION_CLIP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

struct Uniforms {
    total_elements: u32,
    alpha: f32,
    qmax: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    if (params.alpha == 0.0) {
        output_data[idx] = 0.0;
        return;
    }

    let scale = params.alpha / params.qmax;
    let inv_scale = 1.0 / scale;
    let v = input_data[idx];
    let q = clamp(round(v * inv_scale), -params.qmax, params.qmax);
    output_data[idx] = q * scale;
}
"#;

/// CPU reference implementation of calibrated symmetric quantization.
pub fn calibration_clip_cpu(x: &[f32], alpha: f32, qmax: f32) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![0.0f32; x.len()];
    }
    let scale = alpha / qmax;
    let inv_scale = 1.0f32 / scale;
    x.iter()
        .map(|&v| (v * inv_scale).round().clamp(-qmax, qmax) * scale)
        .collect()
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn calibration_clip_auto(x: &[f32], alpha: f32, qmax: f32) -> Vec<f32> {
    calibration_clip_cpu(x, alpha, qmax)
}
