//! CPU and WGSL kernels for Rotation and Incoherence PTQ (QuaRot, SpinQuant, and QuIP#).

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

/// WGSL shader for Fast Walsh-Hadamard Transform (FWHT) butterfly stage:
pub const FWHT_BUTTERFLY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

struct Uniforms {
    total_elements: u32,
    half_step: u32,
    inv_sqrt2: f32,
};
@group(0) @binding(1) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let step = params.half_step * 2u;
    let block = idx / params.half_step;
    let offset = idx % params.half_step;
    let base = block * step + offset;

    if (base + params.half_step >= params.total_elements) {
        return;
    }

    let u = data[base];
    let v = data[base + params.half_step];
    data[base] = (u + v) * params.inv_sqrt2;
    data[base + params.half_step] = (u - v) * params.inv_sqrt2;
}
"#;

/// CPU in-place Fast Walsh-Hadamard Transform (O(d log d)) for incoherence rotation.
pub fn fwht_incoherence_rotation_cpu(data: &mut [f32], signs: &[f32], dim: usize) {
    let n = data.len();
    assert_eq!(n % dim, 0);
    let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

    for row in data.chunks_mut(dim) {
        // 1. Rademacher sign flip
        for (x, &s) in row.iter_mut().zip(signs.iter()) {
            *x *= s;
        }

        // 2. Fast butterfly stages
        let mut half_step = 1;
        while half_step < dim {
            let step = half_step * 2;
            let mut i = 0;
            while i < dim {
                for j in 0..half_step {
                    let u = row[i + j];
                    let v = row[i + j + half_step];
                    row[i + j] = (u + v) * inv_sqrt2;
                    row[i + j + half_step] = (u - v) * inv_sqrt2;
                }
                i += step;
            }
            half_step = step;
        }
    }
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn fwht_incoherence_rotation_auto(data: &mut [f32], signs: &[f32], dim: usize) {
    fwht_incoherence_rotation_cpu(data, signs, dim);
}
