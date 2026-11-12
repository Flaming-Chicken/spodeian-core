//! CPU and WGSL kernels for Direct Preference DPO Ipo KTO SimPO DNO and Spo.

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

/// DPO Logits CPU kernel:
/// logits[b] = beta * ((pi_w[b] - pi_l[b]) - (ref_w[b] - ref_l[b])).
pub fn dpo_logits_cpu(
    pi_w: &[f32],
    pi_l: &[f32],
    ref_w: &[f32],
    ref_l: &[f32],
    batch: usize,
    beta: f32,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; batch];
    for bi in 0..batch {
        let pi_diff = pi_w[bi] - pi_l[bi];
        let ref_diff = ref_w[bi] - ref_l[bi];
        logits[bi] = beta * (pi_diff - ref_diff);
    }
    logits
}

/// DPO Logits WGSL compute shader
pub const DPO_LOGITS_WGSL: &str = r#"
struct Config {
    batch: u32,
    beta: f32,
};

@group(0) @binding(0) var<storage, read> pi_w: array<f32>;
@group(0) @binding(1) var<storage, read> pi_l: array<f32>;
@group(0) @binding(2) var<storage, read> ref_w: array<f32>;
@group(0) @binding(3) var<storage, read> ref_l: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.batch) {
        return;
    }

    let pi_diff = pi_w[bi] - pi_l[bi];
    let ref_diff = ref_w[bi] - ref_l[bi];
    logits[bi] = cfg.beta * (pi_diff - ref_diff);
}
"#;
