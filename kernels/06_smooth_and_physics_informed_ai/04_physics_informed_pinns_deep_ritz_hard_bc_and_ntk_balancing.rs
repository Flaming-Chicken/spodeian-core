//! CPU and WGSL compute kernels for Physics Informed PINNs Deep Ritz Hard Bc and NTK Balancing.

use std::f32::consts::PI;

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

/// Computes Hard Boundary Condition ansatz and Deep Ritz energy density on CPU.
///
/// Returns (u_hard, u_hard_x, ritz_energy).
pub fn hard_bc_deep_ritz_eval_cpu(
    x: &[f32],
    u_raw: &[f32],
    u_x: &[f32],
) -> (Vec<f32>, Vec<f32>, f32) {
    let n = x.len();
    let mut u_hard = vec![0.0f32; n];
    let mut u_hard_x = vec![0.0f32; n];
    let mut energy_sum = 0.0f32;

    for i in 0..n {
        let xi = x[i];
        let ur = u_raw[i];
        let ux = u_x[i];

        let d = xi * (1.0 - xi);
        let d_x = 1.0 - 2.0 * xi;

        let uh = d * ur;
        let uh_x = d_x * ur + d * ux;

        u_hard[i] = uh;
        u_hard_x[i] = uh_x;

        let f = PI * PI * (PI * xi).sin();
        let density = 0.5 * uh_x * uh_x - f * uh;
        energy_sum += density;
    }

    let ritz_energy = energy_sum / (n as f32);
    (u_hard, u_hard_x, ritz_energy)
}

/// WGSL compute shader for parallel Hard BC and Deep Ritz energy density evaluation.
pub const PINN_DEEP_RITZ_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> u_raw: array<f32>;
@group(0) @binding(2) var<storage, read> u_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_hard: array<f32>;
@group(0) @binding(4) var<storage, read_write> u_hard_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> energy_density: array<f32>;

const PI: f32 = 3.141592653589793;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&x);
    if (idx >= n) {
        return;
    }

    let xi = x[idx];
    let ur = u_raw[idx];
    let ux = u_x[idx];

    let d = xi * (1.0 - xi);
    let d_x = 1.0 - 2.0 * xi;

    let uh = d * ur;
    let uh_x = d_x * ur + d * ux;

    u_hard[idx] = uh;
    u_hard_x[idx] = uh_x;

    let f = PI * PI * sin(PI * xi);
    energy_density[idx] = 0.5 * uh_x * uh_x - f * uh;
}
"#;

/// Dispatch wrapper with CPU fallback for hard BC evaluation.
pub fn hard_bc_eval_auto(x: &[f32], u_raw: &[f32], u_x: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
    hard_bc_deep_ritz_eval_cpu(x, u_raw, u_x)
}
