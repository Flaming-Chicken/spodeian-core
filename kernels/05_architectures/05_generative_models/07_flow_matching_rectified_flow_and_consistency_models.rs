//! CPU (and optional WGSL) kernels for Flow Matching Rectified Flow and Consistency Models.

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

/// Conditional Flow Matching straight path interpolation: x_t = (1 - t) * x_0 + t * x_1 and v = x_1 - x_0.
pub fn conditional_flow_matching_cpu(
    x_0: &[f32],
    x_1: &[f32],
    t: &[f32],
    batch: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = batch * dim;
    let mut x_t = vec![0.0f32; total];
    let mut v_target = vec![0.0f32; total];

    for b in 0..batch {
        let t_val = t[b];
        let one_minus_t = 1.0 - t_val;
        for d in 0..dim {
            let idx = b * dim + d;
            x_t[idx] = one_minus_t * x_0[idx] + t_val * x_1[idx];
            v_target[idx] = x_1[idx] - x_0[idx];
        }
    }
    (x_t, v_target)
}

/// Single-step Euler forward integration along vector field v.
pub fn rectified_euler_step_cpu(x: &[f32], v: &[f32], dt: f32) -> Vec<f32> {
    x.iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| xi + dt * vi)
        .collect()
}

/// Consistency skip and output coefficients.
pub fn consistency_coefficients_cpu(
    t: &[f32],
    epsilon: f32,
    sigma_data: f32,
) -> (Vec<f32>, Vec<f32>) {
    let sigma2 = sigma_data * sigma_data;
    let mut c_skip = Vec::with_capacity(t.len());
    let mut c_out = Vec::with_capacity(t.len());

    for &ti in t {
        let dt = ti - epsilon;
        let skip = sigma2 / (dt * dt + sigma2);
        let out = (sigma_data * dt) / (sigma2 + ti * ti).sqrt();
        c_skip.push(skip);
        c_out.push(out);
    }
    (c_skip, c_out)
}

/// WGSL shader for element-wise flow matching interpolation and target velocity generation.
pub const WGSL_FLOW_MATCHING_INTERPOLATION: &str = r#"
@group(0) @binding(0) var<storage, read> x_0: array<f32>;
@group(0) @binding(1) var<storage, read> x_1: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> v_target: array<f32>;

struct Uniforms {
    batch_size: u32,
    dim: u32,
};
@group(0) @binding(5) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.batch_size * u.dim;
    if (idx >= total) {
        return;
    }
    let b = idx / u.dim;
    let t_val = t[b];
    let val_0 = x_0[idx];
    let val_1 = x_1[idx];

    x_t[idx] = (1.0 - t_val) * val_0 + t_val * val_1;
    v_target[idx] = val_1 - val_0;
}
"#;
