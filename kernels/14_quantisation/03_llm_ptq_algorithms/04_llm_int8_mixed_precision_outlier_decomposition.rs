//! CPU and WGSL kernels for LLM.int8() Mixed Precision Outlier Decomposition.

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

/// WGSL shader for scanning activation columns and detecting outlier channels (|x| >= threshold).
pub const OUTLIER_SCAN_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> outlier_flags: array<u32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    threshold: f32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    if (col >= params.d_in) {
        return;
    }
    var max_val = 0.0f;
    for (var b = 0u; b < params.batch; b = b + 1u) {
        let val = abs(x_in[b * params.d_in + col]);
        if (val > max_val) {
            max_val = val;
        }
    }
    if (max_val >= params.threshold) {
        outlier_flags[col] = 1u;
    } else {
        outlier_flags[col] = 0u;
    }
}
"#;

/// CPU reference implementation of outlier channel detection and index extraction.
pub fn llm_int8_scan_outliers_cpu(
    x: &[f32],
    batch: usize,
    d_in: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>) {
    let mut outlier_indices = Vec::new();
    let mut normal_indices = Vec::new();

    for col in 0..d_in {
        let mut max_val = 0.0f32;
        for b in 0..batch {
            let v = x[b * d_in + col].abs();
            if v > max_val {
                max_val = v;
            }
        }
        if max_val >= threshold {
            outlier_indices.push(col);
        } else {
            normal_indices.push(col);
        }
    }

    (outlier_indices, normal_indices)
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn llm_int8_scan_outliers_auto(
    x: &[f32],
    batch: usize,
    d_in: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>) {
    llm_int8_scan_outliers_cpu(x, batch, d_in, threshold)
}
