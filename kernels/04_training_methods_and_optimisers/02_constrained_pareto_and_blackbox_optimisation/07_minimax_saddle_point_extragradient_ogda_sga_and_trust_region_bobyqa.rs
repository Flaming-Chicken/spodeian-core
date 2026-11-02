//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 14:
//! 07_minimax_saddle_point_extragradient_ogda_sga_and_trust_region_bobyqa in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for 2-player bilinear minimax game step.
pub const MINIMAX_STEP_SHADER: &str = r#"
struct MinimaxUniforms {
    len: u32,
    eta_x: f32,
    eta_y: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MinimaxUniforms;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> y_in: array<f32>;
@group(0) @binding(3) var<storage, read> grad_x_in: array<f32>;
@group(0) @binding(4) var<storage, read> grad_y_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> x_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> y_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    x_out[idx] = x_in[idx] - uniforms.eta_x * grad_x_in[idx];
    y_out[idx] = y_in[idx] + uniforms.eta_y * grad_y_in[idx];
}
"#;

fn mat_vec_mul(a: &[f32], v: &[f32]) -> [f32; 2] {
    [a[0] * v[0] + a[1] * v[1], a[2] * v[0] + a[3] * v[1]]
}

fn mat_t_vec_mul(a: &[f32], v: &[f32]) -> [f32; 2] {
    [a[0] * v[0] + a[2] * v[1], a[1] * v[0] + a[3] * v[1]]
}

/// CPU implementation of Korpelevich Extragradient step.
pub fn step_extragradient_cpu(
    x: &[f32],
    y: &[f32],
    a: &[f32],
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    let gx = mat_vec_mul(a, y);
    let gy = mat_t_vec_mul(a, x);

    let x_tilde = [x[0] - eta_x * gx[0], x[1] - eta_x * gx[1]];
    let y_tilde = [y[0] + eta_y * gy[0], y[1] + eta_y * gy[1]];

    let real_gx_tilde = mat_vec_mul(a, &y_tilde);
    let real_gy_tilde = mat_t_vec_mul(a, &x_tilde);

    (
        vec![
            x[0] - eta_x * real_gx_tilde[0],
            x[1] - eta_x * real_gx_tilde[1],
        ],
        vec![
            y[0] + eta_y * real_gy_tilde[0],
            y[1] + eta_y * real_gy_tilde[1],
        ],
    )
}

/// CPU implementation of Optimistic Gradient Descent-Ascent (OGDA).
pub fn step_ogda_cpu(
    x: &[f32],
    y: &[f32],
    x_prev: &[f32],
    y_prev: &[f32],
    a: &[f32],
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    let gx = mat_vec_mul(a, y);
    let gy = mat_t_vec_mul(a, x);
    let gxp = mat_vec_mul(a, y_prev);
    let gyp = mat_t_vec_mul(a, x_prev);

    let x_next = vec![
        x[0] - 2.0 * eta_x * gx[0] + eta_x * gxp[0],
        x[1] - 2.0 * eta_x * gx[1] + eta_x * gxp[1],
    ];
    let y_next = vec![
        y[0] + 2.0 * eta_y * gy[0] - eta_y * gyp[0],
        y[1] + 2.0 * eta_y * gy[1] - eta_y * gyp[1],
    ];

    (x_next, y_next)
}

/// CPU implementation of Symplectic Gradient Adjustment (SGA).
pub fn step_sga_cpu(
    x: &[f32],
    y: &[f32],
    a: &[f32],
    lambda_sga: f32,
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    let at_x = mat_t_vec_mul(a, x);
    let a_at_x = mat_vec_mul(a, &at_x);

    let a_y = mat_vec_mul(a, y);
    let at_a_y = mat_t_vec_mul(a, &a_y);

    let gx = mat_vec_mul(a, y);
    let gy = mat_t_vec_mul(a, x);

    let ux = [
        gx[0] + lambda_sga * a_at_x[0],
        gx[1] + lambda_sga * a_at_x[1],
    ];
    let uy = [
        gy[0] - lambda_sga * at_a_y[0],
        gy[1] - lambda_sga * at_a_y[1],
    ];

    let x_next = vec![x[0] - eta_x * ux[0], x[1] - eta_x * ux[1]];
    let y_next = vec![y[0] + eta_y * uy[0], y[1] + eta_y * uy[1]];

    (x_next, y_next)
}

/// CPU implementation of Nelder-Mead reflection point.
pub fn nelder_mead_reflect_cpu(
    x_best: &[f32],
    x_second: &[f32],
    x_worst: &[f32],
    alpha: f32,
) -> Vec<f32> {
    let n = x_best.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let centroid = 0.5 * (x_best[i] + x_second[i]);
        out[i] = centroid + alpha * (centroid - x_worst[i]);
    }
    out
}

/// Linear anchor forward pass: x @ w + b.
pub fn anchor_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for c in 0..out_dim {
            let mut sum = b[c];
            for k in 0..in_dim {
                sum += x[r * in_dim + k] * w[k * out_dim + c];
            }
            out[r * out_dim + c] = sum;
        }
    }
    out
}

/// Auto-dispatchers with CPU fallback.
pub fn step_extragradient_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    a: &[f32],
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_extragradient_cpu(x, y, a, eta_x, eta_y)
}

pub fn step_ogda_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    x_prev: &[f32],
    y_prev: &[f32],
    a: &[f32],
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_ogda_cpu(x, y, x_prev, y_prev, a, eta_x, eta_y)
}

pub fn step_sga_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    a: &[f32],
    lambda_sga: f32,
    eta_x: f32,
    eta_y: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_sga_cpu(x, y, a, lambda_sga, eta_x, eta_y)
}

pub fn nelder_mead_reflect_auto(
    _device: &KernelDevice,
    x_best: &[f32],
    x_second: &[f32],
    x_worst: &[f32],
    alpha: f32,
) -> Vec<f32> {
    nelder_mead_reflect_cpu(x_best, x_second, x_worst, alpha)
}
