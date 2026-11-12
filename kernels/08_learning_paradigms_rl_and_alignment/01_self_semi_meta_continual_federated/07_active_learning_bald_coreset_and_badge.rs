//! CPU and WGSL kernels for Active Learning Bald Coreset and Badge.

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

/// Fused BADGE gradient embedding CPU kernel:
/// For each sample b, finds argmax class c_star = argmax_c probs[b, c].
/// Computes out[b, c * d + k] = (probs[b, c] - (c == c_star ? 1 : 0)) * feats[b, k].
pub fn badge_gradient_embeddings_cpu(
    feats: &[f32],
    probs: &[f32],
    b: usize,
    d: usize,
    c: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * c * d];
    for bi in 0..b {
        let p_offset = bi * c;
        let f_offset = bi * d;
        let mut max_p = -1.0f32;
        let mut argmax = 0;
        for ci in 0..c {
            let p = probs[p_offset + ci];
            if p > max_p {
                max_p = p;
                argmax = ci;
            }
        }

        let out_b_offset = bi * c * d;
        for ci in 0..c {
            let y = if ci == argmax { 1.0f32 } else { 0.0f32 };
            let diff = probs[p_offset + ci] - y;
            let out_c_offset = out_b_offset + ci * d;
            for di in 0..d {
                out[out_c_offset + di] = diff * feats[f_offset + di];
            }
        }
    }
    out
}

/// Fused BADGE gradient embedding WGSL compute shader
pub const BADGE_GRADIENT_EMBEDDING_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
    c: u32,
};

@group(0) @binding(0) var<storage, read> feats: array<f32>;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let di = global_id.x;
    let bi = global_id.y;

    if (bi >= cfg.b || di >= cfg.d) {
        return;
    }

    let p_offset = bi * cfg.c;
    var max_p: f32 = -1.0;
    var argmax: u32 = 0u;

    for (var ci: u32 = 0u; ci < cfg.c; ci = ci + 1u) {
        let p = probs[p_offset + ci];
        if (p > max_p) {
            max_p = p;
            argmax = ci;
        }
    }

    let f_val = feats[bi * cfg.d + di];
    let out_b_offset = bi * cfg.c * cfg.d;

    for (var ci: u32 = 0u; ci < cfg.c; ci = ci + 1u) {
        var y: f32 = 0.0;
        if (ci == argmax) {
            y = 1.0;
        }
        let diff = probs[p_offset + ci] - y;
        embeddings[out_b_offset + ci * cfg.d + di] = diff * f_val;
    }
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn badge_gradient_embeddings_auto(
    feats: &[f32],
    probs: &[f32],
    b: usize,
    d: usize,
    c: usize,
) -> Vec<f32> {
    badge_gradient_embeddings_cpu(feats, probs, b, d, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_badge_gradient_embeddings_cpu() {
        let feats = vec![1.0, 2.0]; // [1, 2]
        let probs = vec![0.8, 0.2]; // [1, 2], argmax = 0
                                    // diff: [0.8 - 1.0, 0.2 - 0.0] = [-0.2, 0.2]
                                    // embed: [-0.2 * 1, -0.2 * 2, 0.2 * 1, 0.2 * 2] = [-0.2, -0.4, 0.2, 0.4]
        let embeds = badge_gradient_embeddings_cpu(&feats, &probs, 1, 2, 2);
        assert!((embeds[0] - (-0.2)).abs() < 1e-5);
        assert!((embeds[1] - (-0.4)).abs() < 1e-5);
        assert!((embeds[2] - 0.2).abs() < 1e-5);
        assert!((embeds[3] - 0.4).abs() < 1e-5);
    }
}
