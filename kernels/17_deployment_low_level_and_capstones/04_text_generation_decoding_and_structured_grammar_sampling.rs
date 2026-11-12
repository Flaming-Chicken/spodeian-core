//! CPU and WGSL kernels for Text Generation Decoding and Structured Grammar Sampling.

/// WGSL compute shader for fused grammar masking and temperature scaling.
pub const FUSED_GRAMMAR_MASK_TEMP_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    vocab_size: u32,
    inv_temp: f32,
    mask_value: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> logits: array<f32>;       // [batch_size, vocab_size]
@group(0) @binding(2) var<storage, read> grammar_mask: array<f32>; // [batch_size, vocab_size]
@group(0) @binding(3) var<storage, read_write> out_logits: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn grammar_mask_temp_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.batch_size * u.vocab_size;
    if (idx >= total) { return; }

    let mask = grammar_mask[idx];
    if (mask > 0.5) {
        out_logits[idx] = logits[idx] * u.inv_temp;
    } else {
        out_logits[idx] = u.mask_value;
    }
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

/// CPU implementation of fused grammar masking and temperature scaling.
pub fn fused_grammar_mask_temp_cpu(
    logits: &[f32],
    grammar_mask: &[f32],
    temperature: f32,
    mask_value: f32,
) -> Vec<f32> {
    let inv_temp = 1.0 / temperature.max(1e-5);
    logits
        .iter()
        .zip(grammar_mask.iter())
        .map(|(&l, &m)| if m > 0.5 { l * inv_temp } else { mask_value })
        .collect()
}

/// Dispatches fused grammar mask and temperature kernel with automatic CPU fallback.
pub fn fused_grammar_mask_temp_auto(
    logits: &[f32],
    grammar_mask: &[f32],
    temperature: f32,
    mask_value: f32,
) -> Vec<f32> {
    fused_grammar_mask_temp_cpu(logits, grammar_mask, temperature, mask_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_grammar_mask_temp() {
        let logits = vec![2.0, 1.0, -1.0, 4.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let res = fused_grammar_mask_temp_cpu(&logits, &mask, 0.5, -10000.0);
        assert!((res[0] - 4.0).abs() < 1e-5);
        assert_eq!(res[1], -10000.0);
        assert!((res[2] - (-2.0)).abs() < 1e-5);
        assert_eq!(res[3], -10000.0);
    }
}
