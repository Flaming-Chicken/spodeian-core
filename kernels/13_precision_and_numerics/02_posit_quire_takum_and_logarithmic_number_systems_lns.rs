//! CPU (and optional WGSL) kernels for Posit Quire Takum and Logarithmic Number Systems LNS.

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

/// WGSL compute shader for Logarithmic Number System (LNS) multiplication.
pub const LNS_MULTIPLICATION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a_in: array<f32>;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_out: array<f32>;

struct Params {
    total_elements: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn lns_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let a = a_in[idx];
    let b = b_in[idx];

    var sign_out = 1.0;
    if ((a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)) {
        sign_out = -1.0;
    } else if (a == 0.0 || b == 0.0) {
        sign_out = 0.0;
    }

    let log_a = log2(abs(a) + params.eps);
    let log_b = log2(abs(b) + params.eps);
    let log_out = log_a + log_b;

    y_out[idx] = sign_out * exp2(log_out);
}
"#;

/// CPU reference implementation for LNS multiplication.
pub fn lns_multiplication_cpu(a: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| {
            let sign = av.signum() * bv.signum();
            let log_a = (av.abs() + eps).log2();
            let log_b = (bv.abs() + eps).log2();
            sign * (2.0f32).powf(log_a + log_b)
        })
        .collect()
}
