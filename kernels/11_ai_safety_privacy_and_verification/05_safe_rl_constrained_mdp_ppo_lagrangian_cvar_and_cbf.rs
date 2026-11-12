//! CPU (and optional WGSL) kernels for Safe RL Constrained MDP PPO Lagrangian CVaR and CBF.

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

/// WGSL compute shader for Control Barrier Function (CBF) QP safety filtering.
/// Projects nominal control u_nominal onto the safe set half-space: a^T u + b >= 0.
pub const CBF_QP_FILTER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> u_nom: array<f32>;
@group(0) @binding(1) var<storage, read> grad_h: array<f32>;
@group(0) @binding(2) var<storage, read> h_val: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_safe: array<f32>;

struct Params {
    batch: u32,
    d_act: u32,
    alpha_cbf: f32,
    _pad0: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch) {
        return;
    }

    let b_offset = b * params.d_act;
    var a_dot_u: f32 = 0.0;
    var a_norm_sq: f32 = 0.0;

    for (var d: u32 = 0u; d < params.d_act; d = d + 1u) {
        let idx = b_offset + d;
        let a = grad_h[idx];
        let u = u_nom[idx];
        a_dot_u = a_dot_u + a * u;
        a_norm_sq = a_norm_sq + a * a;
    }

    let b_val = params.alpha_cbf * h_val[b];
    let c_viol = a_dot_u + b_val;
    var corr: f32 = 0.0;
    if (c_viol < 0.0) {
        corr = (-c_viol) / (a_norm_sq + 1e-8);
    }

    for (var d: u32 = 0u; d < params.d_act; d = d + 1u) {
        let idx = b_offset + d;
        u_safe[idx] = u_nom[idx] + corr * grad_h[idx];
    }
}
"#;

/// CPU reference implementation of CBF QP safety filtering.
pub fn cbf_qp_filter_cpu(
    u_nominal: &[f32],
    grad_h: &[f32],
    h_val: &[f32],
    alpha_cbf: f32,
    batch: usize,
    d_act: usize,
) -> Vec<f32> {
    let mut u_safe = vec![0.0f32; batch * d_act];
    for b in 0..batch {
        let b_offset = b * d_act;
        let mut a_dot_u = 0.0f32;
        let mut a_norm_sq = 0.0f32;
        for d in 0..d_act {
            let idx = b_offset + d;
            a_dot_u += grad_h[idx] * u_nominal[idx];
            a_norm_sq += grad_h[idx] * grad_h[idx];
        }
        let b_val = alpha_cbf * h_val[b];
        let c_viol = a_dot_u + b_val;
        let corr = (-c_viol).max(0.0) / (a_norm_sq + 1e-8);
        for d in 0..d_act {
            let idx = b_offset + d;
            u_safe[idx] = u_nominal[idx] + corr * grad_h[idx];
        }
    }
    u_safe
}
