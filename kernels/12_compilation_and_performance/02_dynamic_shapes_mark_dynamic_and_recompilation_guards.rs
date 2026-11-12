//! CPU (and optional WGSL) kernels for Dynamic Shapes Mark Dynamic and Recompilation Guards.

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

/// WGSL compute shader for dynamic RMSNorm:
/// y = (x / sqrt(mean(x^2) + eps)) * weight.
pub const DYNAMIC_RMS_NORM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> weight_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out: array<f32>;

struct Params {
    batch: u32,
    d_feat: u32,
    eps: f32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch) {
        return;
    }

    let d = params.d_feat;
    var sum_sq: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let val = x_in[b * d + j];
        sum_sq = sum_sq + val * val;
    }
    let variance = sum_sq / f32(d);
    let inv_rms = 1.0 / sqrt(variance + params.eps);

    for (var j: u32 = 0u; j < d; j = j + 1u) {
        y_out[b * d + j] = (x_in[b * d + j] * inv_rms) * weight_in[j];
    }
}
"#;

/// CPU reference implementation of dynamic RMSNorm.
pub fn dynamic_rms_norm_cpu(
    x: &[f32],
    weight: &[f32],
    batch: usize,
    d_feat: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d_feat];
    for b in 0..batch {
        let mut sum_sq = 0.0f32;
        for j in 0..d_feat {
            let val = x[b * d_feat + j];
            sum_sq += val * val;
        }
        let inv_rms = 1.0f32 / ((sum_sq / d_feat as f32) + eps).sqrt();
        for j in 0..d_feat {
            out[b * d_feat + j] = (x[b * d_feat + j] * inv_rms) * weight[j];
        }
    }
    out
}
