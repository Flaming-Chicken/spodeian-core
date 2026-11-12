//! CPU and WGSL kernels for Expert Parallelism and 3D/4D Parallel Composition.

/// WGSL compute shader for fused MoE expert output combination:
/// `y[t, d] = sum_k (probs[t, k] * expert_outs[t, k, d])`.
pub const FUSED_EXPERT_COMBINE_WGSL: &str = r#"
struct Uniforms {
    num_tokens: u32,
    top_k: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> probs: array<f32>;        // [num_tokens, top_k]
@group(0) @binding(2) var<storage, read> expert_outs: array<f32>;  // [num_tokens, top_k, d_model]
@group(0) @binding(3) var<storage, read_write> y_out: array<f32>;  // [num_tokens, d_model]

@compute @workgroup_size(16, 16, 1)
fn expert_combine_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x; // token index
    let d = gid.y; // feature dimension index

    if (t >= u.num_tokens || d >= u.d_model) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < u.top_k; k++) {
        let p = probs[t * u.top_k + k];
        let val = expert_outs[(t * u.top_k + k) * u.d_model + d];
        acc += p * val;
    }

    y_out[t * u.d_model + d] = acc;
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

/// CPU MoE expert combination kernel: `y[t, d] = sum_k (probs[t, k] * expert_outs[t, k, d])`.
pub fn expert_combine_cpu(
    probs: &[f32],
    expert_outs: &[f32],
    num_tokens: usize,
    top_k: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut y_out = vec![0.0f32; num_tokens * d_model];
    for t in 0..num_tokens {
        for d in 0..d_model {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let p = probs[t * top_k + k];
                let val = expert_outs[(t * top_k + k) * d_model + d];
                acc += p * val;
            }
            y_out[t * d_model + d] = acc;
        }
    }
    y_out
}

/// CPU 3D mesh rank mapping kernel: maps rank to (dp, pp, tp) coordinates.
pub fn mesh_3d_rank_to_coord_cpu(
    rank: usize,
    _dp: usize,
    pp: usize,
    tp: usize,
) -> (usize, usize, usize) {
    let dp_idx = rank / (pp * tp);
    let rem = rank % (pp * tp);
    let pp_idx = rem / tp;
    let tp_idx = rem % tp;
    (dp_idx, pp_idx, tp_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_combine_cpu() {
        let probs = vec![0.6, 0.4]; // 1 token, top-2
        let expert_outs = vec![10.0, 20.0, 30.0, 40.0]; // top-2 outputs of dim 2
        let y = expert_combine_cpu(&probs, &expert_outs, 1, 2, 2);
        // 0.6 * [10, 20] + 0.4 * [30, 40] = [6 + 12, 12 + 16] = [18.0, 28.0]
        assert_eq!(y, vec![18.0, 28.0]);
    }

    #[test]
    fn test_mesh_3d_rank_to_coord_cpu() {
        let coord = mesh_3d_rank_to_coord_cpu(5, 2, 2, 2);
        // 5 / 4 = 1, rem = 1, 1 / 2 = 0, 1 % 2 = 1 -> (1, 0, 1)
        assert_eq!(coord, (1, 0, 1));
    }
}
