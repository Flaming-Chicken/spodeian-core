//! CPU and WGSL kernels for SmoothQuant and Outlier Migration W8A8.

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

/// WGSL shader for fused SmoothQuant activation smoothing: X_smooth[b, j] = X[b, j] / scale[j]
pub const SMOOTHQUANT_SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> x_smooth: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let j = global_id.y;
    if (b >= params.batch || j >= params.d_in) {
        return;
    }
    let idx = b * params.d_in + j;
    x_smooth[idx] = x_in[idx] / scale[j];
}
"#;

/// CPU reference implementation of SmoothQuant activation channel smoothing.
pub fn smoothquant_scale_cpu(x_in: &[f32], scale: &[f32], batch: usize, d_in: usize) -> Vec<f32> {
    let mut x_smooth = vec![0.0f32; batch * d_in];
    for b in 0..batch {
        for j in 0..d_in {
            let idx = b * d_in + j;
            x_smooth[idx] = x_in[idx] / scale[j];
        }
    }
    x_smooth
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn smoothquant_scale_auto(x_in: &[f32], scale: &[f32], batch: usize, d_in: usize) -> Vec<f32> {
    smoothquant_scale_cpu(x_in, scale, batch, d_in)
}
