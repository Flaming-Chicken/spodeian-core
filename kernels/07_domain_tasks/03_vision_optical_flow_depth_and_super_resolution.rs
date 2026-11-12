//! CPU (and optional WGSL) kernels for Vision Optical Flow Depth and Super Resolution.

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

/// Fused Charbonnier loss element-wise evaluation on CPU:
/// out[i] = sqrt( (pred[i] - target[i])^2 + eps^2 )
pub fn charbonnier_loss_cpu(pred: &[f32], target: &[f32], eps: f32) -> Vec<f32> {
    let eps_sq = eps * eps;
    pred.iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            (diff * diff + eps_sq).sqrt()
        })
        .collect()
}

/// WGSL compute shader for parallel element-wise Charbonnier loss evaluation.
pub const CHARBONNIER_LOSS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> pred_buf: array<f32>;
@group(0) @binding(1) var<storage, read> target_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_buf: array<f32>;

struct LossParams {
    total_elements: u32,
    eps_sq: f32,
};
@group(0) @binding(3) var<uniform> params: LossParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.total_elements) {
        return;
    }

    let diff = pred_buf[i] - target_buf[i];
    loss_buf[i] = sqrt(diff * diff + params.eps_sq);
}
"#;
