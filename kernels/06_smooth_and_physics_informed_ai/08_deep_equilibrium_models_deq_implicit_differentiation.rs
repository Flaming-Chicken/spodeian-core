//! CPU (and optional WGSL) kernels for Deep Equilibrium Models DEQ Implicit Differentiation.

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

/// Fixed-point Picard iteration kernel on CPU:
/// z_{k+1} = tanh(z @ w_z + x @ w_x + b)
pub fn deq_picard_step_cpu(
    z: &[f32],
    x: &[f32],
    w_z: &[f32],
    w_x: &[f32],
    b: &[f32],
    batch: usize,
    d_x: usize,
    d_z: usize,
) -> Vec<f32> {
    let mut z_next = vec![0.0f32; batch * d_z];
    for b_idx in 0..batch {
        for o in 0..d_z {
            let mut sum = b[o];
            for i in 0..d_z {
                sum += z[b_idx * d_z + i] * w_z[i * d_z + o];
            }
            for i in 0..d_x {
                sum += x[b_idx * d_x + i] * w_x[i * d_z + o];
            }
            z_next[b_idx * d_z + o] = sum.tanh();
        }
    }
    z_next
}

/// WGSL compute shader for fused parallel Picard fixed point update across batch and dimension.
pub const DEQ_PICARD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> z_buf: array<f32>;
@group(0) @binding(1) var<storage, read> x_buf: array<f32>;
@group(0) @binding(2) var<storage, read> wz_buf: array<f32>;
@group(0) @binding(3) var<storage, read> wx_buf: array<f32>;
@group(0) @binding(4) var<storage, read> b_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> z_next_buf: array<f32>;

struct Dimensions {
    batch: u32,
    d_x: u32,
    d_z: u32,
};
@group(0) @binding(6) var<uniform> dims: Dimensions;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = dims.batch * dims.d_z;
    if (idx >= total_elements) {
        return;
    }

    let b_idx = idx / dims.d_z;
    let o = idx % dims.d_z;

    var sum: f32 = b_buf[o];
    for (var i: u32 = 0u; i < dims.d_z; i = i + 1u) {
        sum = sum + z_buf[b_idx * dims.d_z + i] * wz_buf[i * dims.d_z + o];
    }
    for (var i: u32 = 0u; i < dims.d_x; i = i + 1u) {
        sum = sum + x_buf[b_idx * dims.d_x + i] * wx_buf[i * dims.d_z + o];
    }

    // Tanh activation: (exp(2x) - 1) / (exp(2x) + 1)
    let e2x = exp(2.0 * sum);
    z_next_buf[idx] = (e2x - 1.0) / (e2x + 1.0);
}
"#;
