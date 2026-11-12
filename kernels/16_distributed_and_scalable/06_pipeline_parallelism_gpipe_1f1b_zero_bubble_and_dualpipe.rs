//! CPU and WGSL kernels for Pipeline Parallelism (GPipe, 1F1B, Zero-Bubble, and DualPipe).

/// WGSL compute shader for DualPipe / Zero-Bubble deferred weight gradient accumulation:
/// `grad_w[i, j] += sum_b (x[b, i] * grad_y[b, j])`.
pub const FUSED_DUALPIPE_BW_ACCUMULATE_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    d_in: u32,
    d_out: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [batch_size, d_in]
@group(0) @binding(2) var<storage, read> grad_y: array<f32>;   // [batch_size, d_out]
@group(0) @binding(3) var<storage, read_write> grad_w: array<f32>; // [d_in, d_out]

@compute @workgroup_size(16, 16, 1)
fn dualpipe_bw_accumulate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x; // d_in index
    let col = gid.y; // d_out index

    if (row >= u.d_in || col >= u.d_out) { return; }

    var acc: f32 = 0.0;
    for (var b = 0u; b < u.batch_size; b++) {
        acc += x[b * u.d_in + row] * grad_y[b * u.d_out + col];
    }

    grad_w[row * u.d_out + col] += acc;
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

/// CPU DualPipe B_W weight gradient calculation and accumulation: `grad_w[i, j] += sum_b (x[b, i] * grad_y[b, j])`.
pub fn dualpipe_bw_accumulate_cpu(
    grad_w: &mut [f32],
    x: &[f32],
    grad_y: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) {
    for i in 0..d_in {
        for j in 0..d_out {
            let mut dot = 0.0f32;
            for b in 0..batch {
                dot += x[b * d_in + i] * grad_y[b * d_out + j];
            }
            grad_w[i * d_out + j] += dot;
        }
    }
}

/// CPU DualPipe B_A activation gradient calculation: `grad_x[b, i] = sum_j (grad_y[b, j] * w[i, j])`.
pub fn dualpipe_ba_cpu(
    grad_y: &[f32],
    w: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut grad_x = vec![0.0f32; batch * d_in];
    for b in 0..batch {
        for i in 0..d_in {
            let mut dot = 0.0f32;
            for j in 0..d_out {
                dot += grad_y[b * d_out + j] * w[i * d_out + j];
            }
            grad_x[b * d_in + i] = dot;
        }
    }
    grad_x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dualpipe_bw_accumulate_cpu() {
        let mut grad_w = vec![0.0f32; 4]; // [2, 2]
        let x = vec![1.0, 2.0]; // [1, 2]
        let grad_y = vec![3.0, 4.0]; // [1, 2]
        dualpipe_bw_accumulate_cpu(&mut grad_w, &x, &grad_y, 1, 2, 2);
        assert_eq!(grad_w, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_dualpipe_ba_cpu() {
        let grad_y = vec![1.0, 2.0]; // [1, 2]
        let w = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2] identity
        let grad_x = dualpipe_ba_cpu(&grad_y, &w, 1, 2, 2);
        assert_eq!(grad_x, vec![1.0, 2.0]);
    }
}
