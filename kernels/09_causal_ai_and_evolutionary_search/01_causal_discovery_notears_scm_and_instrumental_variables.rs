//! CPU and WGSL kernels for Causal Discovery NOTEARS SCM and Instrumental Variables.

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

/// WGSL shader for computing NOTEARS acyclicity gradient elementwise:
/// grad_h[r, c] = 2.0 * exp_M[c, r] * W[r, c]
pub const NOTEARS_GRAD_WGSL: &str = r#"
struct Params {
    dim: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> exp_M: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_h: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    let d = params.dim;
    if (r >= d || c >= d) {
        return;
    }
    let idx_rc = r * d + c;
    let idx_cr = c * d + r;
    grad_h[idx_rc] = 2.0 * exp_M[idx_cr] * W[idx_rc];
}
"#;

/// CPU reference calculation of NOTEARS acyclicity gradient.
pub fn notears_grad_cpu(w: &[f32], exp_m: &[f32], d: usize) -> Vec<f32> {
    let mut grad_h = vec![0.0f32; d * d];
    for r in 0..d {
        for c in 0..d {
            let idx_rc = r * d + c;
            let idx_cr = c * d + r;
            grad_h[idx_rc] = 2.0 * exp_m[idx_cr] * w[idx_rc];
        }
    }
    grad_h
}

/// Automatic dispatcher falling back to CPU.
pub fn notears_grad_auto(w: &[f32], exp_m: &[f32], d: usize) -> Vec<f32> {
    notears_grad_cpu(w, exp_m, d)
}
