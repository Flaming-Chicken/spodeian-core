//! CPU (and optional WGSL) kernels for Vision Segmentation Unet Mask2former and SAM Promptable.

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

/// CPU SAM Random Fourier positional prompt feature embedding:
/// points: [num_points, 2] in [0, 1]
/// gaussian_matrix: [d_feat, 2]
/// out: [num_points, 2 * d_feat] where first half is sin, second half is cos.
pub fn sam_fourier_embed_cpu(
    points: &[f32],
    gaussian_matrix: &[f32],
    num_points: usize,
    d_feat: usize,
) -> Vec<f32> {
    let two_pi = 2.0f32 * std::f32::consts::PI;
    let out_dim = 2 * d_feat;
    let mut out = vec![0.0f32; num_points * out_dim];

    for p in 0..num_points {
        let px = points[p * 2];
        let py = points[p * 2 + 1];
        let row_offset = p * out_dim;

        for d in 0..d_feat {
            let bx = gaussian_matrix[d * 2];
            let by = gaussian_matrix[d * 2 + 1];
            let proj = two_pi * (px * bx + py * by);
            out[row_offset + d] = proj.sin();
            out[row_offset + d_feat + d] = proj.cos();
        }
    }
    out
}

/// WGSL compute shader for parallel SAM Fourier prompt feature projection.
pub const SAM_FOURIER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> points_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_matrix_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> embed_buf: array<f32>;

struct FourierParams {
    num_points: u32,
    d_feat: u32,
};
@group(0) @binding(3) var<uniform> params: FourierParams;

const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = global_id.x;
    if (p >= params.num_points) {
        return;
    }

    let px = points_buf[p * 2u];
    let py = points_buf[p * 2u + 1u];
    let out_dim = 2u * params.d_feat;
    let row_offset = p * out_dim;

    for (var d: u32 = 0u; d < params.d_feat; d = d + 1u) {
        let bx = b_matrix_buf[d * 2u];
        let by = b_matrix_buf[d * 2u + 1u];
        let proj = TWO_PI * (px * bx + py * by);
        embed_buf[row_offset + d] = sin(proj);
        embed_buf[row_offset + params.d_feat + d] = cos(proj);
    }
}
"#;
