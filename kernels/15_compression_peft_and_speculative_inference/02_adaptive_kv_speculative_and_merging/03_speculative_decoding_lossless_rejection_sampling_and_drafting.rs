//! CPU and WGSL kernels for Speculative Decoding Lossless Rejection Sampling and Drafting.

/// WGSL compute shader for speculative probability ratio and acceptance mask evaluation.
pub const SPECULATIVE_VERIFICATION_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    gamma: u32,
    vocab: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> target_probs: array<f32>; // [batch, gamma, vocab]
@group(0) @binding(2) var<storage, read> draft_probs: array<f32>; // [batch, gamma, vocab]
@group(0) @binding(3) var<storage, read> draft_tokens: array<f32>; // [batch, gamma]
@group(0) @binding(4) var<storage, read> uniform_rands: array<f32>; // [batch, gamma]
@group(0) @binding(5) var<storage, read_write> accepted_mask: array<f32>; // [batch, gamma]

@compute @workgroup_size(64, 1, 1)
fn speculative_verify_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch) { return; }

    var all_prev_accepted = true;

    for (var i = 0u; i < u.gamma; i++) {
        let t_id = u32(draft_tokens[b * u.gamma + i]);
        let prob_offset = (b * u.gamma + i) * u.vocab + t_id;
        let p_val = target_probs[prob_offset];
        let q_val = max(draft_probs[prob_offset], 1e-8);
        let ratio = min(1.0, p_val / q_val);
        let rand_val = uniform_rands[b * u.gamma + i];

        if (all_prev_accepted && rand_val <= ratio) {
            accepted_mask[b * u.gamma + i] = 1.0;
        } else {
            accepted_mask[b * u.gamma + i] = 0.0;
            all_prev_accepted = false;
        }
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

/// CPU sequential speculative rejection step.
pub fn speculative_rejection_cpu(
    target_probs: &[f32],
    draft_probs: &[f32],
    draft_tokens: &[f32],
    uniform_rands: &[f32],
    batch: usize,
    gamma: usize,
    vocab: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut accepted_mask = vec![0.0f32; batch * gamma];
    let mut num_accepted = vec![0.0f32; batch];

    for b in 0..batch {
        let mut count = 0.0f32;
        for i in 0..gamma {
            let t_id = draft_tokens[b * gamma + i] as usize;
            let offset = (b * gamma + i) * vocab + t_id;
            let p_val = target_probs[offset];
            let q_val = draft_probs[offset].max(1e-8);
            let ratio = (p_val / q_val).min(1.0);

            if uniform_rands[b * gamma + i] <= ratio {
                accepted_mask[b * gamma + i] = 1.0;
                count += 1.0;
            } else {
                break;
            }
        }
        num_accepted[b] = count;
    }

    (accepted_mask, num_accepted)
}
