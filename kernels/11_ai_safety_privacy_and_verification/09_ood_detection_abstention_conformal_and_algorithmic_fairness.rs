//! CPU (and optional WGSL) kernels for OOD Detection Abstention Conformal and Algorithmic Fairness.

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

/// WGSL compute shader for Helmholtz Free Energy OOD scoring:
/// E(x) = -T * log(sum_c exp(logits[c] / T)).
pub const ENERGY_OOD_SCORE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> logits_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> energy_out: array<f32>;

struct Params {
    batch: u32,
    num_classes: u32,
    temperature: f32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch) {
        return;
    }

    let c_num = params.num_classes;
    var max_val: f32 = -1e30;
    for (var c: u32 = 0u; c < c_num; c = c + 1u) {
        let val = logits_in[b * c_num + c] / params.temperature;
        if (val > max_val) {
            max_val = val;
        }
    }

    var sum_exp: f32 = 0.0;
    for (var c: u32 = 0u; c < c_num; c = c + 1u) {
        sum_exp = sum_exp + exp((logits_in[b * c_num + c] / params.temperature) - max_val);
    }

    let lse = max_val + log(sum_exp);
    energy_out[b] = -params.temperature * lse;
}
"#;

/// CPU reference implementation of Helmholtz Free Energy OOD score.
pub fn energy_ood_score_cpu(
    logits: &[f32],
    batch: usize,
    num_classes: usize,
    temperature: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch];
    for b in 0..batch {
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = logits[b * num_classes + c] / temperature;
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            sum_exp += ((logits[b * num_classes + c] / temperature) - max_val).exp();
        }
        let lse = max_val + sum_exp.ln();
        out[b] = -temperature * lse;
    }
    out
}
