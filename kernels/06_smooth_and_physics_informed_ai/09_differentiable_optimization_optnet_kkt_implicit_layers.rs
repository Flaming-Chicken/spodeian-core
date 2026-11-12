//! CPU (and optional WGSL) kernels for Differentiable Optimization OptNet KKT Implicit Layers.

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

/// Fused log-barrier gradient descent step for QP inequality constraints on CPU:
/// z_{k+1} = z_k - eta * (Q z_k + p + sum_i (g_i / (t * max(h_i - g_i^T z_k, eps))))
pub fn optnet_barrier_step_cpu(
    z: &[f32],
    q: &[f32],
    p: &[f32],
    g: &[f32],
    h_bound: &[f32],
    batch: usize,
    d: usize,
    k_constraints: usize,
    eta: f32,
    t_barrier: f32,
) -> Vec<f32> {
    let mut z_next = vec![0.0f32; batch * d];
    for b in 0..batch {
        // Compute slacks s_i = h_i - sum_j g[i, j] * z[b, j]
        let mut lambda = vec![0.0f32; k_constraints];
        for i in 0..k_constraints {
            let mut gz = 0.0f32;
            for j in 0..d {
                gz += g[i * d + j] * z[b * d + j];
            }
            let slack = (h_bound[b * k_constraints + i] - gz).max(1e-3f32);
            lambda[i] = 1.0f32 / (t_barrier * slack);
        }

        for j in 0..d {
            let mut qz = 0.0f32;
            for i in 0..d {
                qz += q[i * d + j] * z[b * d + i];
            }
            let mut grad_barrier = 0.0f32;
            for i in 0..k_constraints {
                grad_barrier += lambda[i] * g[i * d + j];
            }
            let grad_total = qz + p[b * d + j] + grad_barrier;
            z_next[b * d + j] = z[b * d + j] - eta * grad_total;
        }
    }
    z_next
}

/// WGSL compute shader for fused parallel log-barrier gradient step across batch and state dimensions.
pub const OPTNET_BARRIER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> z_buf: array<f32>;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> p_buf: array<f32>;
@group(0) @binding(3) var<storage, read> g_buf: array<f32>;
@group(0) @binding(4) var<storage, read> h_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> z_next_buf: array<f32>;

struct BarrierParams {
    batch: u32,
    d: u32,
    k_constraints: u32,
    eta: f32,
    t_barrier: f32,
};
@group(0) @binding(6) var<uniform> params: BarrierParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch * params.d;
    if (idx >= total) {
        return;
    }

    let b = idx / params.d;
    let j = idx % params.d;

    // Q * z
    var qz: f32 = 0.0;
    for (var i: u32 = 0u; i < params.d; i = i + 1u) {
        qz = qz + q_buf[i * params.d + j] * z_buf[b * params.d + i];
    }

    // Barrier gradient: sum_i g[i, j] / (t * max(h[i] - g_i^T z, 1e-3))
    var grad_barrier: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k_constraints; i = i + 1u) {
        var gz: f32 = 0.0;
        for (var col: u32 = 0u; col < params.d; col = col + 1u) {
            gz = gz + g_buf[i * params.d + col] * z_buf[b * params.d + col];
        }
        let slack = max(h_buf[b * params.k_constraints + i] - gz, 1e-3);
        let lambda_i = 1.0 / (params.t_barrier * slack);
        grad_barrier = grad_barrier + lambda_i * g_buf[i * params.d + j];
    }

    let grad_total = qz + p_buf[b * params.d + j] + grad_barrier;
    z_next_buf[idx] = z_buf[idx] - params.eta * grad_total;
}
"#;
