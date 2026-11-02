//! CPU (and optional WGSL) kernels for Vision Detection Anchor Free FCOS and DETR.

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

/// Fused FCOS Centerness calculation on CPU:
/// sqrt( min(l, r)/max(l, r) * min(t, b)/max(t, b) )
pub fn fcos_centerness_cpu(ltrb: &[f32], total_points: usize) -> Vec<f32> {
    let mut centerness = vec![0.0f32; total_points];
    for i in 0..total_points {
        let l = ltrb[i * 4];
        let t = ltrb[i * 4 + 1];
        let r = ltrb[i * 4 + 2];
        let b = ltrb[i * 4 + 3];

        let lr_min = l.min(r);
        let lr_max = l.max(r).max(1e-6f32);
        let tb_min = t.min(b);
        let tb_max = t.max(b).max(1e-6f32);

        let prod = ((lr_min / lr_max) * (tb_min / tb_max)).clamp(0.0f32, 1.0f32);
        centerness[i] = prod.sqrt();
    }
    centerness
}

/// WGSL compute shader for fused parallel FCOS centerness evaluation across spatial detection grid.
pub const FCOS_CENTERNESS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ltrb_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> centerness_buf: array<f32>;

struct GridDims {
    total_points: u32,
};
@group(0) @binding(2) var<uniform> dims: GridDims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= dims.total_points) {
        return;
    }

    let l = ltrb_buf[i * 4u];
    let t = ltrb_buf[i * 4u + 1u];
    let r = ltrb_buf[i * 4u + 2u];
    let b = ltrb_buf[i * 4u + 3u];

    let lr_min = min(l, r);
    let lr_max = max(max(l, r), 1e-6);
    let tb_min = min(t, b);
    let tb_max = max(max(t, b), 1e-6);

    let prod = clamp((lr_min / lr_max) * (tb_min / tb_max), 0.0, 1.0);
    centerness_buf[i] = sqrt(prod);
}
"#;
