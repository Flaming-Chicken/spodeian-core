//! CPU (and optional WGSL) kernels for INT UINT Fixed Point Q Format Dyadic and RNS Arithmetic.

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

/// WGSL compute shader for integer-only dyadic rescaling.
pub const DYADIC_RESCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> acc_in: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params {
    total_elements: u32,
    m0: i32,
    shift: i32,
    _pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn dyadic_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    // In 32-bit shader math: emulate normalized multiplication
    let val = f32(acc_in[idx]);
    let mult = f32(params.m0) * exp2(-f32(params.shift));
    out[idx] = floor(val * mult + 0.5);
}
"#;

/// CPU reference implementation for dyadic rescaling.
pub fn dyadic_rescale_cpu(acc_int: &[i32], m0: i64, shift: i64) -> Vec<f32> {
    let final_shift = shift - 31;
    acc_int
        .iter()
        .map(|&v| {
            let acc_64 = v as i64;
            let scaled = (acc_64 * m0 + (1i64 << 30)) >> 31;
            let out = if final_shift > 0 {
                (scaled + (1i64 << (final_shift - 1))) >> final_shift
            } else {
                scaled << (-final_shift)
            };
            out as f32
        })
        .collect()
}
