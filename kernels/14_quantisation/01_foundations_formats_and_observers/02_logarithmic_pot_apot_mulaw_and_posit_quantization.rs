//! CPU (and optional WGSL) kernels for Logarithmic PoT APoT μ-Law and Posit Quantization.

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

/// WGSL compute shader for Additive Power-of-Two (APoT) quantization.
pub const APOT_QUANTIZE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> x_out: array<f32>;

struct Params {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

const LEVELS: array<f32, 8> = array<f32, 8>(
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0
);

@compute @workgroup_size(64)
fn apot_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let v = x_in[idx];
    let sign = select(-1.0, 1.0, v >= 0.0);
    let abs_v = clamp(abs(v), 0.0, 1.0);

    var best_level = LEVELS[0];
    var min_diff = abs(abs_v - LEVELS[0]);

    for (var i: u32 = 1u; i < 8u; i = i + 1u) {
        let diff = abs(abs_v - LEVELS[i]);
        if (diff < min_diff) {
            min_diff = diff;
            best_level = LEVELS[i];
        }
    }

    x_out[idx] = sign * best_level;
}
"#;

/// CPU reference implementation for APoT quantization.
pub fn apot_quantize_cpu(x: &[f32]) -> Vec<f32> {
    let levels = [0.0f32, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0];
    x.iter()
        .map(|&v| {
            let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
            let abs_v = v.abs().clamp(0.0, 1.0);
            let mut best_level = levels[0];
            let mut min_diff = (abs_v - levels[0]).abs();

            for &lvl in &levels[1..] {
                let diff = (abs_v - lvl).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_level = lvl;
                }
            }
            sign * best_level
        })
        .collect()
}
