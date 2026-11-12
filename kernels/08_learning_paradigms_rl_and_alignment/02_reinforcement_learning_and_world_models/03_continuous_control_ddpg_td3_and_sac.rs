//! CPU and WGSL kernels for Continuous Control Ddpg Td3 and SAC.

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

/// Squashed Gaussian action and tanh Jacobian diagonal correction CPU kernel:
/// a[b, d] = tanh(u[b, d])
/// log_det[b] = sum_d ln(max(1 - a[b, d]^2, eps))
pub fn squashed_gaussian_jacobian_cpu(
    u: &[f32],
    b: usize,
    d: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut actions = vec![0.0f32; b * d];
    let mut log_det = vec![0.0f32; b];

    for bi in 0..b {
        let mut jac = 0.0f32;
        let offset = bi * d;
        for di in 0..d {
            let a = u[offset + di].tanh();
            actions[offset + di] = a;
            jac += (1.0f32 - a * a).max(eps).ln();
        }
        log_det[bi] = jac;
    }
    (actions, log_det)
}

/// Squashed Gaussian action and Jacobian WGSL compute shader
pub const SQUASHED_GAUSSIAN_JACOBIAN_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read_write> actions: array<f32>;
@group(0) @binding(2) var<storage, read_write> log_det: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.b) {
        return;
    }

    let offset = bi * cfg.d;
    var sum_jac: f32 = 0.0;

    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        let val = u[offset + di];
        let a = tanh(val);
        actions[offset + di] = a;
        let diff = max(1.0 - a * a, cfg.eps);
        sum_jac = sum_jac + log(diff);
    }

    log_det[bi] = sum_jac;
}
"#;
