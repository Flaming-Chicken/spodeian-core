//! CPU (and optional WGSL) kernels for FP8 E4M3 E5M2 FNUZ and Minifloats FP6 FP4.

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

/// WGSL compute shader for FP8 E4M3FN simulation (4 exp, 3 mantissa, bias 7, max 448.0).
pub const FP8_E4M3_QUANTIZE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_out: array<f32>;

struct Params {
    total_elements: u32,
    inv_scale: f32,
    scale: f32,
    max_val: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn fp8_e4m3_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let val = x_in[idx] * params.inv_scale;
    if (val == 0.0) {
        x_out[idx] = 0.0;
        return;
    }

    let sign = select(-1.0, 1.0, val > 0.0);
    let abs_val = abs(val);
    let e = floor(log2(abs_val));
    let e_min_normal = -6.0; // 1 - 7

    var step: f32;
    if (e < e_min_normal) {
        step = exp2(e_min_normal - 3.0); // 2^(-9)
    } else {
        step = exp2(e - 3.0);
    }

    let q = min(round(abs_val / step) * step, params.max_val);
    x_out[idx] = sign * q * params.scale;
}
"#;

/// CPU reference implementation for FP8 E4M3FN quantisation.
pub fn fp8_e4m3fn_quantize_cpu(x: &[f32], scale: f32) -> Vec<f32> {
    let inv_scale = 1.0f32 / scale;
    x.iter()
        .map(|&v| {
            if v == 0.0 {
                return 0.0;
            }
            let val = v * inv_scale;
            let sign = if val < 0.0 { -1.0 } else { 1.0 };
            let abs_v = val.abs();
            let e = abs_v.log2().floor();
            let e_min_normal = -6.0f32;
            let step = if e < e_min_normal {
                (2.0f32).powf(-9.0)
            } else {
                (2.0f32).powf(e - 3.0)
            };
            let q = ((abs_v / step).round() * step).min(448.0);
            sign * q * scale
        })
        .collect()
}
