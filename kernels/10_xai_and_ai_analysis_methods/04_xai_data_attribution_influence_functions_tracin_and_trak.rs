//! CPU (and optional WGSL) kernels for XAI Data Attribution Influence Functions TracIn and TRAK.

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

/// WGSL compute shader for TracIn checkpoint gradient dot-product accumulation.
pub const TRACIN_ACCUMULATION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> train_grads: array<f32>;
@group(0) @binding(1) var<storage, read> test_grads: array<f32>;
@group(0) @binding(2) var<storage, read> lrs: array<f32>;
@group(0) @binding(3) var<storage, read_write> step_contributions: array<f32>;

struct Params {
    num_checkpoints: u32,
    num_params: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.num_checkpoints) {
        return;
    }
    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < params.num_params; d = d + 1u) {
        let idx = t * params.num_params + d;
        dot = dot + train_grads[idx] * test_grads[idx];
    }
    step_contributions[t] = lrs[t] * dot;
}
"#;

/// CPU reference implementation of TracIn gradient descent attribution.
pub fn tracin_attribution_cpu(
    train_grads: &[f32],
    test_grads: &[f32],
    lrs: &[f32],
    num_checkpoints: usize,
    num_params: usize,
) -> f32 {
    let mut total = 0.0f32;
    for t in 0..num_checkpoints {
        let mut dot = 0.0f32;
        for d in 0..num_params {
            let idx = t * num_params + d;
            dot += train_grads[idx] * test_grads[idx];
        }
        total += lrs[t] * dot;
    }
    total
}
