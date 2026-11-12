//! CPU (and optional WGSL) kernels for Heterophilic, Temporal, Cellular Sheaves, and Knowledge Graph Embeddings.

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

/// Continuous cellular sheaf diffusion step on CPU: H_next = H - dt * (L_F @ H).
pub fn sheaf_diffusion_step_cpu(
    h: &[f32],
    sheaf_laplacian: &[f32],
    num_nodes: usize,
    dim: usize,
    dt: f32,
) -> Vec<f32> {
    let mut out = h.to_vec();
    for i in 0..num_nodes {
        for d in 0..dim {
            let mut diff = 0.0f32;
            for j in 0..num_nodes {
                diff += sheaf_laplacian[i * num_nodes + j] * h[j * dim + d];
            }
            out[i * dim + d] -= dt * diff;
        }
    }
    out
}

/// RotatE Knowledge Graph Embedding score on CPU.
pub fn rotate_kge_score_cpu(
    h_re: &[f32],
    h_im: &[f32],
    theta: &[f32],
    t_re: &[f32],
    t_im: &[f32],
    batch: usize,
    dim: usize,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(batch);

    for b in 0..batch {
        let mut dist_sq = 0.0f32;
        for d in 0..dim {
            let idx = b * dim + d;
            let cos_th = theta[idx].cos();
            let sin_th = theta[idx].sin();

            let rot_re = h_re[idx] * cos_th - h_im[idx] * sin_th;
            let rot_im = h_re[idx] * sin_th + h_im[idx] * cos_th;

            let diff_re = rot_re - t_re[idx];
            let diff_im = rot_im - t_im[idx];

            dist_sq += diff_re * diff_re + diff_im * diff_im;
        }
        scores.push(-(dist_sq + 1e-12).sqrt());
    }
    scores
}

/// WGSL shader for parallel RotatE complex Hadamard rotation and distance reduction.
pub const WGSL_ROTATE_KGE: &str = r#"
@group(0) @binding(0) var<storage, read> h_re: array<f32>;
@group(0) @binding(1) var<storage, read> h_im: array<f32>;
@group(0) @binding(2) var<storage, read> theta: array<f32>;
@group(0) @binding(3) var<storage, read> t_re: array<f32>;
@group(0) @binding(4) var<storage, read> t_im: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;

struct Uniforms {
    batch_size: u32,
    dim: u32,
};
@group(0) @binding(6) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch_size) {
        return;
    }
    var dist_sq: f32 = 0.0;
    for (var d: u32 = 0u; d < u.dim; d = d + 1u) {
        let idx = b * u.dim + d;
        let cos_th = cos(theta[idx]);
        let sin_th = sin(theta[idx]);

        let rot_re = h_re[idx] * cos_th - h_im[idx] * sin_th;
        let rot_im = h_re[idx] * sin_th + h_im[idx] * cos_th;

        let diff_re = rot_re - t_re[idx];
        let diff_im = rot_im - t_im[idx];

        dist_sq = dist_sq + diff_re * diff_re + diff_im * diff_im;
    }
    scores[b] = -sqrt(dist_sq + 1e-12);
}
"#;
