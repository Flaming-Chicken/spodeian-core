//! CPU and WGSL kernels for Collectives and DDP Gradient Bucketing Mechanics.

/// WGSL compute shader for fused DDP bucket accumulation and averaging across ranks.
pub const FUSED_DDP_BUCKET_ACCUMULATE_WGSL: &str = r#"
struct Uniforms {
    world_size: u32,
    bucket_size: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> rank_grads: array<f32>; // [world_size, bucket_size]
@group(0) @binding(2) var<storage, read_write> averaged_grads: array<f32>; // [bucket_size]

@compute @workgroup_size(64, 1, 1)
fn ddp_bucket_accumulate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.bucket_size) { return; }

    var acc: f32 = 0.0;
    for (var r = 0u; r < u.world_size; r++) {
        acc += rank_grads[r * u.bucket_size + idx];
    }
    averaged_grads[idx] = acc / f32(u.world_size);
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

/// CPU DDP bucket accumulation and averaging: `avg[i] = sum_r rank_grads[r, i] / world_size`.
pub fn ddp_bucket_accumulate_cpu(
    rank_grads: &[f32],
    world_size: usize,
    bucket_size: usize,
) -> Vec<f32> {
    let mut averaged = vec![0.0f32; bucket_size];
    let inv_w = 1.0f32 / (world_size as f32);

    for i in 0..bucket_size {
        let mut sum = 0.0f32;
        for r in 0..world_size {
            sum += rank_grads[r * bucket_size + i];
        }
        averaged[i] = sum * inv_w;
    }

    averaged
}
