//! CPU (and optional WGSL) kernels for Optimisation as Architecture LISTA ADMM-Net CRATE and Mesa-Optimizers.

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

/// Fused LISTA step on CPU:
/// z_{k+1} = sign(lin) * max(|lin| - theta, 0), where lin = z_k W_s + y W_e
pub fn lista_step_cpu(
    z: &[f32],
    y: &[f32],
    w_s: &[f32],
    w_e: &[f32],
    theta: &[f32],
    batch: usize,
    m_meas: usize,
    n_sparse: usize,
) -> Vec<f32> {
    let mut z_next = vec![0.0f32; batch * n_sparse];
    for b in 0..batch {
        for j in 0..n_sparse {
            let mut lin = 0.0f32;
            for i in 0..n_sparse {
                lin += z[b * n_sparse + i] * w_s[i * n_sparse + j];
            }
            for i in 0..m_meas {
                lin += y[b * m_meas + i] * w_e[i * n_sparse + j];
            }
            let th = theta[j];
            let abs_val = lin.abs();
            let sign = if lin > 0.0 {
                1.0f32
            } else if lin < 0.0 {
                -1.0f32
            } else {
                0.0f32
            };
            z_next[b * n_sparse + j] = sign * (abs_val - th).max(0.0f32);
        }
    }
    z_next
}

/// WGSL compute shader for fused parallel LISTA step across batch and sparse dimensions.
pub const LISTA_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> z_buf: array<f32>;
@group(0) @binding(1) var<storage, read> y_buf: array<f32>;
@group(0) @binding(2) var<storage, read> ws_buf: array<f32>;
@group(0) @binding(3) var<storage, read> we_buf: array<f32>;
@group(0) @binding(4) var<storage, read> theta_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> z_next_buf: array<f32>;

struct ListaDims {
    batch: u32,
    m_meas: u32,
    n_sparse: u32,
};
@group(0) @binding(6) var<uniform> dims: ListaDims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = dims.batch * dims.n_sparse;
    if (idx >= total) {
        return;
    }

    let b = idx / dims.n_sparse;
    let j = idx % dims.n_sparse;

    var lin: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.n_sparse; i = i + 1u) {
        lin = lin + z_buf[b * dims.n_sparse + i] * ws_buf[i * dims.n_sparse + j];
    }
    for (var i: u32 = 0u; i < dims.m_meas; i = i + 1u) {
        lin = lin + y_buf[b * dims.m_meas + i] * we_buf[i * dims.n_sparse + j];
    }

    let th = theta_buf[j];
    let abs_val = abs(lin);
    let s = sign(lin);
    z_next_buf[idx] = s * max(abs_val - th, 0.0);
}
"#;
