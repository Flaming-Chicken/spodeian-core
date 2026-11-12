//! CPU and WGSL kernels for Semi Supervised Fixmatch Mean Teacher and Nnpu Learning.

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

/// Fused FixMatch confidence thresholding and cross-entropy CPU kernel:
/// For each sample in batch B:
/// 1. Finds max_p = max_k weak_probs[b, k] and argmax = argmax_k weak_probs[b, k].
/// 2. If max_p >= tau, returns -strong_log_probs[b, argmax], else 0.
pub fn fixmatch_threshold_ce_cpu(
    weak_probs: &[f32],
    strong_log_probs: &[f32],
    b: usize,
    k: usize,
    tau: f32,
) -> Vec<f32> {
    let mut loss = vec![0.0f32; b];
    for i in 0..b {
        let offset = i * k;
        let mut max_p = -1.0f32;
        let mut argmax = 0;
        for j in 0..k {
            let p = weak_probs[offset + j];
            if p > max_p {
                max_p = p;
                argmax = j;
            }
        }
        if max_p >= tau {
            loss[i] = -strong_log_probs[offset + argmax];
        }
    }
    loss
}

/// Fused FixMatch confidence thresholding WGSL compute shader
pub const FIXMATCH_THRESHOLD_CE_WGSL: &str = r#"
struct Config {
    b: u32,
    k: u32,
    tau: f32,
};

@group(0) @binding(0) var<storage, read> weak_probs: array<f32>;
@group(0) @binding(1) var<storage, read> strong_log_probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.b) {
        return;
    }

    let offset = i * cfg.k;
    var max_p: f32 = -1.0;
    var argmax: u32 = 0u;

    for (var j: u32 = 0u; j < cfg.k; j = j + 1u) {
        let p = weak_probs[offset + j];
        if (p > max_p) {
            max_p = p;
            argmax = j;
        }
    }

    if (max_p >= cfg.tau) {
        loss[i] = -strong_log_probs[offset + argmax];
    } else {
        loss[i] = 0.0;
    }
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn fixmatch_threshold_ce_auto(
    weak_probs: &[f32],
    strong_log_probs: &[f32],
    b: usize,
    k: usize,
    tau: f32,
) -> Vec<f32> {
    fixmatch_threshold_ce_cpu(weak_probs, strong_log_probs, b, k, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixmatch_threshold_ce_cpu() {
        let weak_p = vec![0.1, 0.9, 0.8, 0.2]; // [2, 2], both >= 0.7
        let strong_lp = vec![-2.0, -0.1, -0.2, -1.8];
        let loss = fixmatch_threshold_ce_cpu(&weak_p, &strong_lp, 2, 2, 0.7);
        assert!((loss[0] - 0.1).abs() < 1e-5);
        assert!((loss[1] - 0.2).abs() < 1e-5);
    }
}
