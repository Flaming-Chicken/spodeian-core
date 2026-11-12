//! CPU and WGSL compute kernels for Numerical Integration (PCEC, RK4, Neural ODE, LTC, and SDE).

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

/// Fused CPU RK4 combination step:
/// x_next = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
pub fn rk4_step_cpu(
    x0: &[f32],
    k1: &[f32],
    k2: &[f32],
    k3: &[f32],
    k4: &[f32],
    h: f32,
) -> Vec<f32> {
    let n = x0.len();
    let mut x_out = vec![0.0f32; n];
    let sixth_h = h / 6.0;

    for i in 0..n {
        let sum_k = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i];
        x_out[i] = x0[i] + sixth_h * sum_k;
    }
    x_out
}

/// Fused CPU Neural SDE Milstein integration step:
/// x_em = x0 + h * f0 + sigma * x0 * dW
/// x_milstein = x_em + 0.5 * sigma^2 * x0 * (dW^2 - h)
pub fn sde_milstein_step_cpu(
    x0: &[f32],
    f0: &[f32],
    dw: &[f32],
    sigma: f32,
    h: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = x0.len();
    let mut x_em = vec![0.0f32; n];
    let mut x_mil = vec![0.0f32; n];
    let half_sig_sq = 0.5 * sigma * sigma;

    for i in 0..n {
        let em = x0[i] + h * f0[i] + sigma * x0[i] * dw[i];
        let mil = em + half_sig_sq * x0[i] * (dw[i] * dw[i] - h);
        x_em[i] = em;
        x_mil[i] = mil;
    }
    (x_em, x_mil)
}

/// WGSL compute shader for parallel RK4 stage aggregation.
pub const RK4_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x0: array<f32>;
@group(0) @binding(1) var<storage, read> k1: array<f32>;
@group(0) @binding(2) var<storage, read> k2: array<f32>;
@group(0) @binding(3) var<storage, read> k3: array<f32>;
@group(0) @binding(4) var<storage, read> k4: array<f32>;
@group(0) @binding(5) var<storage, read_write> x_out: array<f32>;

struct Uniforms {
    h: f32,
    total_elements: u32,
};
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    let sixth_h = params.h / 6.0;
    let sum_k = k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx];
    x_out[idx] = x0[idx] + sixth_h * sum_k;
}
"#;

/// Dispatch wrapper with CPU fallback for RK4 step.
pub fn rk4_eval_auto(
    x0: &[f32],
    k1: &[f32],
    k2: &[f32],
    k3: &[f32],
    k4: &[f32],
    h: f32,
) -> Vec<f32> {
    rk4_step_cpu(x0, k1, k2, k3, k4, h)
}
