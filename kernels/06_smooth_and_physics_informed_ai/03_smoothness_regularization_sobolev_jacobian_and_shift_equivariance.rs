//! CPU (and optional WGSL) kernels for Smoothness Regularization Sobolev Jacobian and Shift Equivariance.

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

/// CPU kernel for 1D anti-aliased BlurPool downsampling with stride 2 and padding 1.
pub fn blur_downsample_1d_cpu(
    x: &[f32],
    batch: usize,
    channels: usize,
    length: usize,
    k0: f32,
    k1: f32,
    k2: f32,
) -> Vec<f32> {
    let out_len = length / 2;
    let mut out = vec![0.0f32; batch * channels * out_len];

    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * length;
            let out_offset = (b * channels + c) * out_len;
            for step in 0..out_len {
                let center = (step * 2) as isize;
                let v0 = if center - 1 < 0 {
                    0.0
                } else {
                    x[in_offset + (center - 1) as usize]
                };
                let v1 = x[in_offset + center as usize];
                let v2 = if center + 1 >= length as isize {
                    0.0
                } else {
                    x[in_offset + (center + 1) as usize]
                };
                out[out_offset + step] = v0 * k0 + v1 * k1 + v2 * k2;
            }
        }
    }
    out
}

/// WGSL shader for fused anti-aliasing 1D blur downsampling.
pub const BLUR_1D_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

struct Uniforms {
    batch: u32,
    channels: u32,
    in_len: u32,
    out_len: u32,
    k0: f32,
    k1: f32,
    k2: f32,
};
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = u.batch * u.channels * u.out_len;
    if (idx >= total) {
        return;
    }
    let step = idx % u.out_len;
    let bc = idx / u.out_len;
    let in_offset = bc * u.in_len;
    let center = i32(step * 2u);

    var v0: f32 = 0.0;
    if (center - 1 >= 0) {
        v0 = x[in_offset + u32(center - 1)];
    }
    let v1: f32 = x[in_offset + u32(center)];
    var v2: f32 = 0.0;
    if (u32(center + 1) < u.in_len) {
        v2 = x[in_offset + u32(center + 1)];
    }
    y[idx] = v0 * u.k0 + v1 * u.k1 + v2 * u.k2;
}
"#;
