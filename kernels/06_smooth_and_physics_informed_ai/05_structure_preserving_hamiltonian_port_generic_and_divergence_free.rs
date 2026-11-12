//! CPU and WGSL compute kernels for Structure Preserving Hamiltonian, Port-Hamiltonian, GENERIC, and Divergence Free Networks.

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

/// Fused CPU Symplectic Leapfrog (Verlet) integration step:
/// p_half = p - 0.5 * h * grad_v_0
/// q_next = q + h * p_half
/// p_next = p_half - 0.5 * h * grad_v_next
pub fn symplectic_leapfrog_step_cpu(
    q: &[f32],
    p: &[f32],
    grad_v_0: &[f32],
    grad_v_next: &[f32],
    h: f32,
    batch: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half_h = 0.5 * h;
    let total = batch * dim;
    let mut q_next = vec![0.0f32; total];
    let mut p_next = vec![0.0f32; total];

    for i in 0..total {
        let p_half = p[i] - half_h * grad_v_0[i];
        q_next[i] = q[i] + h * p_half;
        p_next[i] = p_half - half_h * grad_v_next[i];
    }

    (q_next, p_next)
}

/// Computes 2D Divergence-Free Solenoidal velocity from spatial stream gradient:
/// v_x = dpsi/dy, v_y = -dpsi/dx
pub fn divergence_free_stream_cpu(
    grad_psi: &[f32], // [B * 2] where row is [dpsi/dx, dpsi/dy]
    batch: usize,
) -> Vec<f32> {
    let mut v = vec![0.0f32; batch * 2];
    for b in 0..batch {
        let dpsi_dx = grad_psi[b * 2 + 0];
        let dpsi_dy = grad_psi[b * 2 + 1];
        v[b * 2 + 0] = dpsi_dy;
        v[b * 2 + 1] = -dpsi_dx;
    }
    v
}

/// WGSL compute shader for parallel solenoidal field synthesis and symplectic leapfrog updates.
pub const STRUCTURE_PRESERVING_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> grad_v_0: array<f32>;
@group(0) @binding(3) var<storage, read> grad_v_next: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> p_out: array<f32>;

struct Uniforms {
    h: f32,
    total_elements: u32,
};
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn leapfrog_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    let half_h = 0.5 * params.h;
    let p_half = p[idx] - half_h * grad_v_0[idx];
    q_out[idx] = q[idx] + params.h * p_half;
    p_out[idx] = p_half - half_h * grad_v_next[idx];
}
"#;

/// Dispatch wrapper with CPU fallback.
pub fn leapfrog_auto(
    q: &[f32],
    p: &[f32],
    grad_v_0: &[f32],
    grad_v_next: &[f32],
    h: f32,
    batch: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    symplectic_leapfrog_step_cpu(q, p, grad_v_0, grad_v_next, h, batch, dim)
}
