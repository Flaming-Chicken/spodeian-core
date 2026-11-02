//! CPU and WGSL kernels for Quantile and Information Theoretic NF4 SF4 FP4 Lloyd Max and Dynamic Tree.

pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

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

/// WGSL shader for NormalFloat4 (NF4) codebook lookup and dequantization.
pub const NF4_QUANTIZE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> codebook: array<f32, 16>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<storage, read_write> block_scales: array<f32>;

struct Uniforms {
    total_elements: u32,
    block_size: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    let block_idx = idx / params.block_size;
    let scale = block_scales[block_idx];
    if (scale == 0.0) {
        output_data[idx] = 0.0;
        return;
    }

    let norm_val = input_data[idx] / scale;
    var best_val = codebook[0];
    var min_diff = abs(norm_val - codebook[0]);

    for (var i: u32 = 1u; i < 16u; i = i + 1u) {
        let diff = abs(norm_val - codebook[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_val = codebook[i];
        }
    }

    output_data[idx] = best_val * scale;
}
"#;

/// CPU reference implementation for NF4 block quantization and dequantization.
pub fn nf4_quantize_block_cpu(x: &[f32], block_size: usize) -> (Vec<f32>, Vec<f32>) {
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
            scales.push(amax);
            let inv_scale = 1.0f32 / amax;
            for &v in chunk {
                let norm_v = v * inv_scale;
                let mut best = NF4_LEVELS[0];
                let mut min_diff = (norm_v - best).abs();
                for &lvl in &NF4_LEVELS[1..] {
                    let diff = (norm_v - lvl).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        best = lvl;
                    }
                }
                dequant.push(best * amax);
            }
        }
    }
    (dequant, scales)
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn nf4_quantize_block_auto(x: &[f32], block_size: usize) -> (Vec<f32>, Vec<f32>) {
    nf4_quantize_block_cpu(x, block_size)
}
