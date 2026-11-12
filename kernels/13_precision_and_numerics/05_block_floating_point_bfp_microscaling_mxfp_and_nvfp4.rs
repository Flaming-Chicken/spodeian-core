//! CPU (and optional WGSL) kernels for Block Floating Point BFP Microscaling MXFP and NVFP4.

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

/// WGSL compute shader for OCP MXFP4 element dequantization using block scale factor.
pub const OCP_MXFP4_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> elem_in: array<f32>;
@group(0) @binding(1) var<storage, read> block_scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> elem_out: array<f32>;

struct Params {
    total_elements: u32,
    block_size: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn mxfp4_dequant_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let block_idx = idx / params.block_size;
    let scale = block_scales[block_idx];
    elem_out[idx] = elem_in[idx] * scale;
}
"#;

/// CPU reference implementation for OCP MXFP4 quantize.
pub fn ocp_mxfp4_quantize_cpu(x: &[f32], block_size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut dequant = Vec::with_capacity(x.len());
    let mut scales = Vec::new();

    for chunk in x.chunks(block_size) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if amax == 0.0 {
            scales.push(1.0f32);
            for _ in 0..chunk.len() {
                dequant.push(0.0f32);
            }
        } else {
            let exp_scale = (amax / 6.0).log2().ceil();
            let scale_block = (2.0f32).powf(exp_scale);
            scales.push(scale_block);

            for &v in chunk {
                let scaled = v / scale_block;
                let sign = if scaled < 0.0 { -1.0 } else { 1.0 };
                let abs_v = scaled.abs();
                let e = abs_v.log2().floor();
                let step = if e < 0.0 {
                    0.5f32
                } else {
                    (2.0f32).powf(e - 1.0)
                };
                let q = ((abs_v / step).round() * step).min(6.0);
                dequant.push(sign * q * scale_block);
            }
        }
    }

    (dequant, scales)
}
