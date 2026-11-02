//! CPU (and optional WGSL) kernels for Koopman Operators SINDy Equation Discovery & 4D-Var Data Assimilation.

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

/// SINDy 2D polynomial + trigonometric feature library on CPU:
/// [1, x1, x2, x1^2, x1*x2, x2^2, sin(x1), sin(x2)] for each row
pub fn sindy_library_cpu(x: &[f32], batch: usize) -> Vec<f32> {
    let mut theta = vec![0.0f32; batch * 8];
    for b in 0..batch {
        let x1 = x[b * 2];
        let x2 = x[b * 2 + 1];
        let offset = b * 8;
        theta[offset] = 1.0f32;
        theta[offset + 1] = x1;
        theta[offset + 2] = x2;
        theta[offset + 3] = x1 * x1;
        theta[offset + 4] = x1 * x2;
        theta[offset + 5] = x2 * x2;
        theta[offset + 6] = x1.sin();
        theta[offset + 7] = x2.sin();
    }
    theta
}

/// WGSL compute shader for parallel generation of SINDy candidate function library.
pub const SINDY_LIBRARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> theta_buf: array<f32>;

struct LibraryDims {
    batch: u32,
};
@group(0) @binding(2) var<uniform> dims: LibraryDims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= dims.batch) {
        return;
    }

    let x1 = x_buf[b * 2u];
    let x2 = x_buf[b * 2u + 1u];
    let offset = b * 8u;

    theta_buf[offset] = 1.0;
    theta_buf[offset + 1u] = x1;
    theta_buf[offset + 2u] = x2;
    theta_buf[offset + 3u] = x1 * x1;
    theta_buf[offset + 4u] = x1 * x2;
    theta_buf[offset + 5u] = x2 * x2;
    theta_buf[offset + 6u] = sin(x1);
    theta_buf[offset + 7u] = sin(x2);
}
"#;
