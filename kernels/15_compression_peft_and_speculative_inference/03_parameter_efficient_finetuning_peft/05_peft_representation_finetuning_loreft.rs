//! CPU and WGSL kernels for PEFT Representation Finetuning Loreft.

/// WGSL compute shader for fused LoReFT intervention: h_out = h + mask * R^T (W h + b - R h).
pub const FUSED_LOREFT_INTERVENE_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    seq_len: u32,
    dim: u32,
    rank: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> h: array<f32>; // [batch, seq_len, dim]
@group(0) @binding(2) var<storage, read> r_mat: array<f32>; // [rank, dim]
@group(0) @binding(3) var<storage, read> w_mat: array<f32>; // [rank, dim]
@group(0) @binding(4) var<storage, read> b_vec: array<f32>; // [rank]
@group(0) @binding(5) var<storage, read> mask: array<f32>; // [batch, seq_len]
@group(0) @binding(6) var<storage, read_write> h_out: array<f32>; // [batch, seq_len, dim]

@compute @workgroup_size(16, 16, 1)
fn loreft_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x; // b * seq_len + s
    let col = gid.y; // d in [0, dim)

    let total_tokens = u.batch * u.seq_len;
    if (token_idx >= total_tokens || col >= u.dim) { return; }

    let base_idx = token_idx * u.dim + col;
    let m = mask[token_idx];
    if (m == 0.0) {
        h_out[base_idx] = h[base_idx];
        return;
    }

    var delta_out: f32 = 0.0;
    for (var r = 0u; r < u.rank; r++) {
        var proj_w: f32 = b_vec[r];
        var proj_r: f32 = 0.0;
        for (var k = 0u; k < u.dim; k++) {
            let h_val = h[token_idx * u.dim + k];
            proj_w += h_val * w_mat[r * u.dim + k];
            proj_r += h_val * r_mat[r * u.dim + k];
        }
        let delta_s = proj_w - proj_r;
        delta_out += delta_s * r_mat[r * u.dim + col];
    }

    h_out[base_idx] = h[base_idx] + m * delta_out;
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

/// CPU LoReFT intervention kernel: `h_out = h + mask * R^T (W h + b - R h)`.
pub fn loreft_intervene_cpu(
    h: &[f32],
    r_mat: &[f32],
    w_mat: &[f32],
    b_vec: &[f32],
    mask: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
    rank: usize,
) -> Vec<f32> {
    let total_elements = batch * seq_len * dim;
    let mut h_out = vec![0.0f32; total_elements];

    for b in 0..batch {
        for s in 0..seq_len {
            let token_idx = b * seq_len + s;
            let m = mask[token_idx];
            let token_offset = token_idx * dim;

            if m == 0.0 {
                for d in 0..dim {
                    h_out[token_offset + d] = h[token_offset + d];
                }
                continue;
            }

            let mut delta_s = vec![0.0f32; rank];
            for r in 0..rank {
                let mut proj_w = b_vec[r];
                let mut proj_r = 0.0f32;
                for k in 0..dim {
                    let h_val = h[token_offset + k];
                    proj_w += h_val * w_mat[r * dim + k];
                    proj_r += h_val * r_mat[r * dim + k];
                }
                delta_s[r] = proj_w - proj_r;
            }

            for d in 0..dim {
                let mut delta_out = 0.0f32;
                for r in 0..rank {
                    delta_out += delta_s[r] * r_mat[r * dim + d];
                }
                h_out[token_offset + d] = h[token_offset + d] + m * delta_out;
            }
        }
    }

    h_out
}
