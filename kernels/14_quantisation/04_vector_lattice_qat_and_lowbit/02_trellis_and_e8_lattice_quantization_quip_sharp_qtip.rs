//! CPU and WGSL kernels for Trellis and E8 Lattice Quantization QuIP# QTIP.

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

/// WGSL compute shader for fused E8 lattice dequantization:
/// `W_deq[n, i] = lattice_points[n, i] * scale`
pub const E8_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> lattice_points: array<f32>;
@group(0) @binding(1) var<storage, read_write> w_deq: array<f32>;

struct Uniforms {
    n_vecs: u32,
    scale: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.n_vecs * 8u;
    if (idx >= total_elements) {
        return;
    }
    w_deq[idx] = lattice_points[idx] * params.scale;
}
"#;

/// CPU reference implementation of fused E8 lattice dequantization.
pub fn e8_dequant_cpu(lattice_points: &[f32], scale: f32, n_vecs: usize) -> Vec<f32> {
    let total = n_vecs * 8;
    let mut w_deq = vec![0.0f32; total];
    for i in 0..total {
        w_deq[i] = lattice_points[i] * scale;
    }
    w_deq
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn e8_dequant_auto(lattice_points: &[f32], scale: f32, n_vecs: usize) -> Vec<f32> {
    e8_dequant_cpu(lattice_points, scale, n_vecs)
}
