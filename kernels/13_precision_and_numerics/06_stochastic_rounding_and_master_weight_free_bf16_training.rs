//! CPU (and optional WGSL) kernels for Stochastic Rounding and Master Weight Free BF16 Training.

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

/// WGSL compute shader for unbiased Stochastic Rounding from FP32 to BF16.
pub const STOCHASTIC_ROUNDING_BF16_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> noise_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> x_out: array<f32>;

struct Params {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn sr_bf16_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let v = x_in[idx];
    if (v == 0.0) {
        x_out[idx] = 0.0;
        return;
    }

    let sign = select(-1.0, 1.0, v > 0.0);
    let abs_v = abs(v);
    let e = floor(log2(abs_v));
    let ulp = exp2(e - 7.0);

    let step_ratio = abs_v / ulp;
    let floor_ratio = floor(step_ratio);
    let frac = step_ratio - floor_ratio;

    let n = noise_in[idx];
    let q_mag = select(floor_ratio * ulp, (floor_ratio + 1.0) * ulp, n < frac);
    x_out[idx] = sign * q_mag;
}
"#;

/// CPU reference implementation for Stochastic Rounding to BF16.
pub fn stochastic_rounding_bf16_cpu(x: &[f32], noise: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(noise.iter())
        .map(|(&v, &n)| {
            if v == 0.0 {
                return 0.0f32;
            }
            let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
            let abs_v = v.abs();
            let e = abs_v.log2().floor();
            let ulp = (2.0f32).powf(e - 7.0);
            let step_ratio = abs_v / ulp;
            let floor_ratio = step_ratio.floor();
            let frac = step_ratio - floor_ratio;
            let q_mag = if n < frac {
                (floor_ratio + 1.0) * ulp
            } else {
                floor_ratio * ulp
            };
            sign * q_mag
        })
        .collect()
}
