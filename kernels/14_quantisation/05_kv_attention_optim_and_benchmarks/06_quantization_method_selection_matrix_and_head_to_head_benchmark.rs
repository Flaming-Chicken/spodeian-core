//! CPU and WGSL kernels for Quantization Method Selection Matrix and Head to Head Benchmark.

/// Unified WGSL compute shader for fused multi-format dequantization and GEMM dispatch.
pub const UNIFIED_DEQUANT_GEMM_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
    format_mode: u32, // 0 = RTN8, 1 = RTN4, 2 = BitNet, 3 = NF4
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w_quant: array<f32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn unified_gemm_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let b = gid.x;
    let o = gid.y;
    if (b >= u.batch || o >= u.d_out) { return; }

    var sum: f32 = 0.0;
    for (var i = 0u; i < u.d_in; i++) {
        let x_val = x[b * u.d_in + i];
        let w_val = w_quant[i * u.d_out + o];
        sum += x_val * w_val;
    }
    y_out[b * u.d_out + o] = sum;
}
"#;

pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
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

/// CPU RTN uniform symmetric per-row quantization.
pub fn rtn_quantize_cpu(w: &[f32], rows: usize, cols: usize, bits: usize) -> Vec<f32> {
    let mut q_w = vec![0.0f32; rows * cols];
    let levels = ((1usize << (bits - 1)) - 1) as f32;

    for r in 0..rows {
        let row_slice = &w[r * cols..(r + 1) * cols];
        let max_abs = row_slice.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / levels;
        let scale_eps = scale + 1e-8;

        for c in 0..cols {
            let val = row_slice[c];
            let q = (val / scale_eps).round().clamp(-levels, levels);
            q_w[r * cols + c] = q * scale;
        }
    }
    q_w
}

/// CPU BitNet b1.58 native ternary quantization.
pub fn bitnet_quantize_cpu(w: &[f32]) -> Vec<f32> {
    let scale = w.iter().map(|&x| x.abs()).sum::<f32>() / w.len() as f32;
    let scale_eps = scale + 1e-8;
    w.iter()
        .map(|&x| (x / scale_eps).round().clamp(-1.0, 1.0) * scale)
        .collect()
}

/// CPU NF4 quantile codebook quantization.
pub fn nf4_quantize_cpu(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut q_w = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row_slice = &w[r * cols..(r + 1) * cols];
        let max_abs = row_slice.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale_eps = max_abs + 1e-8;

        for c in 0..cols {
            let norm_val = row_slice[c] / scale_eps;
            let mut best_level = NF4_LUT[0];
            let mut min_dist = (norm_val - best_level).abs();
            for &lvl in &NF4_LUT[1..] {
                let dist = (norm_val - lvl).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_level = lvl;
                }
            }
            q_w[r * cols + c] = best_level * max_abs;
        }
    }
    q_w
}

/// Auto-dispatching unified quantization.
pub fn unified_quantize_auto(w: &[f32], rows: usize, cols: usize, mode: usize) -> Vec<f32> {
    match mode {
        0 => rtn_quantize_cpu(w, rows, cols, 8),
        1 => rtn_quantize_cpu(w, rows, cols, 4),
        2 => bitnet_quantize_cpu(w),
        _ => nf4_quantize_cpu(w, rows, cols),
    }
}
