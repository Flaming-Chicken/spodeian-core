//! CPU (and optional WGSL) kernels for AI Brain Surgery Knowledge Editing ROME MEMIT and Stitching.

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

/// WGSL compute shader for ROME rank-one weight editing update.
pub const ROME_RANK_ONE_UPDATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> u_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_new: array<f32>;

struct Params {
    d_out: u32,
    d_in: u32,
    denom: f32,
    _pad0: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.d_out * params.d_in;
    if (idx >= total) {
        return;
    }
    let r = idx / params.d_in;
    let c = idx % params.d_in;
    let delta = (residual[r] * u_vec[c]) / params.denom;
    w_new[idx] = weight[idx] + delta;
}
"#;

/// CPU reference implementation of ROME rank-one weight update.
pub fn rome_update_cpu(
    weight: &[f32],
    residual: &[f32],
    u_vec: &[f32],
    denom: f32,
    d_out: usize,
    d_in: usize,
) -> Vec<f32> {
    let total = d_out * d_in;
    let mut w_new = vec![0.0f32; total];
    for r in 0..d_out {
        for c in 0..d_in {
            let idx = r * d_in + c;
            let delta = (residual[r] * u_vec[c]) / denom;
            w_new[idx] = weight[idx] + delta;
        }
    }
    w_new
}
