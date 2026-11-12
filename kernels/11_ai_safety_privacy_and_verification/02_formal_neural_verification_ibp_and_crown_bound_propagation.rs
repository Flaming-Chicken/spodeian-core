//! CPU (and optional WGSL) kernels for Formal Neural Verification IBP and CROWN Bound Propagation.

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

/// WGSL compute shader for CROWN ReLU Linear Relaxation.
/// Evaluates upper and lower linear relaxation parameters (s_L, b_L, s_U, b_U) across all neurons.
pub const CROWN_RELU_RELAXATION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> l_in: array<f32>;
@group(0) @binding(1) var<storage, read> u_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_l_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_l_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_u_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> b_u_out: array<f32>;

struct Params {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let l = l_in[idx];
    let u = u_in[idx];

    if (l >= 0.0) {
        s_l_out[idx] = 1.0;
        b_l_out[idx] = 0.0;
        s_u_out[idx] = 1.0;
        b_u_out[idx] = 0.0;
    } else if (u <= 0.0) {
        s_l_out[idx] = 0.0;
        b_l_out[idx] = 0.0;
        s_u_out[idx] = 0.0;
        b_u_out[idx] = 0.0;
    } else {
        let diff = max(u - l, 1e-8);
        s_u_out[idx] = u / diff;
        b_u_out[idx] = -(u * l) / diff;

        var sl_val: f32 = 0.0;
        if (u > -l) {
            sl_val = 1.0;
        }
        s_l_out[idx] = sl_val;
        b_l_out[idx] = 0.0;
    }
}
"#;

/// CPU reference implementation of CROWN ReLU linear relaxation.
pub fn crown_relu_relaxation_cpu(
    l_in: &[f32],
    u_in: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = l_in.len();
    let mut s_l = vec![0.0f32; n];
    let mut b_l = vec![0.0f32; n];
    let mut s_u = vec![0.0f32; n];
    let mut b_u = vec![0.0f32; n];

    for i in 0..n {
        let l = l_in[i];
        let u = u_in[i];

        if l >= 0.0 {
            s_l[i] = 1.0;
            b_l[i] = 0.0;
            s_u[i] = 1.0;
            b_u[i] = 0.0;
        } else if u <= 0.0 {
            s_l[i] = 0.0;
            b_l[i] = 0.0;
            s_u[i] = 0.0;
            b_u[i] = 0.0;
        } else {
            let diff = (u - l).max(1e-8);
            s_u[i] = u / diff;
            b_u[i] = -(u * l) / diff;

            s_l[i] = if u > -l { 1.0 } else { 0.0 };
            b_l[i] = 0.0;
        }
    }

    (s_l, b_l, s_u, b_u)
}
