//! CPU (and optional WGSL) kernels for Adversarial Robustness FGSM PGD TRADES GCG and Smoothing.

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

/// WGSL compute shader for Projected Gradient Descent (PGD) step in L_infinity ball.
pub const PGD_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_adv: array<f32>;
@group(0) @binding(1) var<storage, read> x_orig: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_adv: array<f32>;

struct Params {
    total_elements: u32,
    alpha: f32,
    eps: f32,
    x_min: f32,
    x_max: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let g = grad[idx];
    var s: f32 = 0.0;
    if (g > 0.0) {
        s = 1.0;
    } else if (g < 0.0) {
        s = -1.0;
    }

    let step = x_adv[idx] + params.alpha * s;
    let proj = clamp(step, x_orig[idx] - params.eps, x_orig[idx] + params.eps);
    out_adv[idx] = clamp(proj, params.x_min, params.x_max);
}
"#;

/// CPU reference implementation of Projected Gradient Descent step.
pub fn pgd_step_cpu(
    x_adv: &[f32],
    x_orig: &[f32],
    grad: &[f32],
    alpha: f32,
    eps: f32,
    x_min: f32,
    x_max: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x_adv.len()];
    for i in 0..x_adv.len() {
        let sign = if grad[i] > 0.0 {
            1.0
        } else if grad[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        let step = x_adv[i] + alpha * sign;
        let proj = step.max(x_orig[i] - eps).min(x_orig[i] + eps);
        out[i] = proj.max(x_min).min(x_max);
    }
    out
}
