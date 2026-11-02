//! CPU (and optional WGSL) kernels for Machine Unlearning SISA Influence Scrub NPO and RMU.

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

/// WGSL compute shader for Negative Preference Optimization (NPO) per-sample loss.
pub const NPO_LOSS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> log_pi_theta: array<f32>;
@group(0) @binding(1) var<storage, read> log_pi_ref: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_out: array<f32>;

struct Params {
    total_elements: u32,
    beta: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let diff = log_pi_theta[idx] - log_pi_ref[idx];
    let z = params.beta * diff;

    var sp: f32 = 0.0;
    if (z > 30.0) {
        sp = z;
    } else if (z < -30.0) {
        sp = exp(z);
    } else {
        sp = log(1.0 + exp(z));
    }

    loss_out[idx] = (2.0 / params.beta) * sp;
}
"#;

/// CPU reference implementation of NPO elementwise loss.
pub fn npo_loss_cpu(log_pi_theta: &[f32], log_pi_ref: &[f32], beta: f32) -> Vec<f32> {
    let n = log_pi_theta.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let diff = log_pi_theta[i] - log_pi_ref[i];
        let z = beta * diff;
        let sp = if z > 30.0 {
            z
        } else if z < -30.0 {
            z.exp()
        } else {
            (1.0 + z.exp()).ln()
        };
        out[i] = (2.0 / beta) * sp;
    }
    out
}
