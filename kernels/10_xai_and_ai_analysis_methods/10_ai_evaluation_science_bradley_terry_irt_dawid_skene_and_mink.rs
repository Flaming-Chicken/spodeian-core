//! CPU (and optional WGSL) kernels for AI Evaluation Science Bradley Terry IRT Dawid Skene and Min-K%.

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

/// WGSL compute shader for Bradley-Terry-Luce win probability matrix: P_{ij} = sigmoid(s_i - s_j).
pub const BTL_WIN_PROBABILITY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> skills: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;

struct Params {
    m: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let m = params.m;
    if (idx >= m * m) {
        return;
    }
    let i = idx / m;
    let j = idx % m;

    let diff = skills[i] - skills[j];
    probs[idx] = 1.0 / (1.0 + exp(-diff));
}
"#;

/// CPU reference implementation of Bradley-Terry-Luce win probability matrix.
pub fn btl_win_probability_cpu(skills: &[f32]) -> Vec<f32> {
    let m = skills.len();
    let mut probs = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let diff = skills[i] - skills[j];
            probs[i * m + j] = 1.0f32 / (1.0f32 + (-diff).exp());
        }
    }
    probs
}
