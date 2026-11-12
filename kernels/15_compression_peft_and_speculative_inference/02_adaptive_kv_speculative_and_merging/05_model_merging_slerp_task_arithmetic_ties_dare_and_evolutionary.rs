//! CPU and WGSL kernels for Model Merging Slerp Task Arithmetic TIES DARE and Evolutionary.

/// WGSL compute shader for parallel task vector accumulation and linear combination.
pub const TASK_ARITHMETIC_ACCUMULATE_WGSL: &str = r#"
struct Uniforms {
    total_elements: u32,
    num_tasks: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> base: array<f32>; // [total_elements]
@group(0) @binding(2) var<storage, read> deltas: array<f32>; // [num_tasks, total_elements]
@group(0) @binding(3) var<storage, read> weights: array<f32>; // [num_tasks]
@group(0) @binding(4) var<storage, read_write> merged: array<f32>; // [total_elements]

@compute @workgroup_size(64, 1, 1)
fn task_arithmetic_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) { return; }

    var val = base[idx];
    for (var k = 0u; k < u.num_tasks; k++) {
        val += weights[k] * deltas[k * u.total_elements + idx];
    }
    merged[idx] = val;
}
"#;

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

/// CPU SLERP interpolation.
pub fn slerp_cpu(v0: &[f32], v1: &[f32], t: f32) -> Vec<f32> {
    let n = v0.len();
    let norm0: f32 = v0.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm1: f32 = v1.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm0 < 1e-6 || norm1 < 1e-6 {
        return (0..n).map(|i| (1.0 - t) * v0[i] + t * v1[i]).collect();
    }

    let dot: f32 = (0..n).map(|i| (v0[i] / norm0) * (v1[i] / norm1)).sum();
    let theta = dot.clamp(-1.0, 1.0).acos();

    if theta.sin().abs() < 1e-6 || theta.abs() < 1e-6 {
        return (0..n).map(|i| (1.0 - t) * v0[i] + t * v1[i]).collect();
    }

    let sin_theta = theta.sin();
    let w0 = ((1.0 - t) * theta).sin() / sin_theta;
    let w1 = (t * theta).sin() / sin_theta;

    (0..n).map(|i| w0 * v0[i] + w1 * v1[i]).collect()
}

/// CPU DARE element-wise drop and rescale.
pub fn dare_rescale_cpu(delta: &[f32], mask: &[f32], drop_prob: f32) -> Vec<f32> {
    let scale = 1.0 / (1.0 - drop_prob);
    delta
        .iter()
        .zip(mask.iter())
        .map(|(&d, &m)| d * m * scale)
        .collect()
}
