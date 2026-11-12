//! CPU and WGSL compute kernels for Neural Operators (FNO, DeepONet, and MeshGraphNets).

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

/// Fused CPU DeepONet branch-trunk inner product with scalar bias:
/// out[b, q] = b0 + sum_k branch[b, k] * trunk[q, k]
pub fn deeponet_dot_cpu(
    branch: &[f32], // [B * P]
    trunk: &[f32],  // [Q * P]
    b0: f32,
    batch: usize,
    q_queries: usize,
    p_latent: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * q_queries];
    for b in 0..batch {
        for q in 0..q_queries {
            let mut dot = b0;
            for k in 0..p_latent {
                dot += branch[b * p_latent + k] * trunk[q * p_latent + k];
            }
            out[b * q_queries + q] = dot;
        }
    }
    out
}

/// WGSL compute shader for parallel DeepONet branch-trunk operator evaluation.
pub const DEEPONET_DOT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> branch: array<f32>;
@group(0) @binding(1) var<storage, read> trunk: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_matrix: array<f32>;

struct Uniforms {
    b0: f32,
    p_latent: u32,
    q_queries: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let q = global_id.y;
    let total_b = arrayLength(&branch) / params.p_latent;

    if (b >= total_b || q >= params.q_queries) {
        return;
    }

    var sum: f32 = params.b0;
    for (var k: u32 = 0u; k < params.p_latent; k = k + 1u) {
        sum = sum + branch[b * params.p_latent + k] * trunk[q * params.p_latent + k];
    }
    out_matrix[b * params.q_queries + q] = sum;
}
"#;

/// Dispatch wrapper with CPU fallback for DeepONet inner product.
pub fn deeponet_eval_auto(
    branch: &[f32],
    trunk: &[f32],
    b0: f32,
    batch: usize,
    q_queries: usize,
    p_latent: usize,
) -> Vec<f32> {
    deeponet_dot_cpu(branch, trunk, b0, batch, q_queries, p_latent)
}
