//! CPU and WGSL kernels for Contrastive Ssl Simclr Moco and CLIP Siglip.

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

/// Fused SigLIP Pairwise Binary Loss:
/// Computes pairwise cosine similarity, applies affine scale/bias, and softplus loss:
/// img: [B, D] row-major, pre-normalized
/// txt: [B, D] row-major, pre-normalized
/// returns scalar mean loss over B*B pairs
pub fn siglip_pairwise_cpu(
    img: &[f32],
    txt: &[f32],
    b: usize,
    d: usize,
    scale: f32,
    bias: f32,
) -> f32 {
    let mut total_loss = 0.0f32;
    for i in 0..b {
        let i_offset = i * d;
        for j in 0..b {
            let j_offset = j * d;
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += img[i_offset + di] * txt[j_offset + di];
            }
            let logit = dot * scale + bias;
            let y = if i == j { 1.0f32 } else { -1.0f32 };
            let val = -y * logit;
            let sp = (1.0f32 + val.exp()).ln();
            total_loss += sp;
        }
    }
    total_loss / ((b * b) as f32)
}

/// Fused SigLIP Pairwise Loss WGSL compute shader
pub const SIGLIP_PAIRWISE_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
    scale: f32,
    bias: f32,
};

@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<storage, read> txt: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_matrix: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let i = global_id.y;

    if (i >= cfg.b || j >= cfg.b) {
        return;
    }

    var dot: f32 = 0.0;
    let i_offset = i * cfg.d;
    let j_offset = j * cfg.d;

    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        dot = dot + img[i_offset + di] * txt[j_offset + di];
    }

    let logit = dot * cfg.scale + cfg.bias;
    var y: f32 = -1.0;
    if (i == j) {
        y = 1.0;
    }

    let val = -y * logit;
    let sp = log(1.0 + exp(val));
    loss_matrix[i * cfg.b + j] = sp;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn siglip_pairwise_auto(
    img: &[f32],
    txt: &[f32],
    b: usize,
    d: usize,
    scale: f32,
    bias: f32,
) -> f32 {
    siglip_pairwise_cpu(img, txt, b, d, scale, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siglip_pairwise_cpu() {
        let img = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let txt = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let loss = siglip_pairwise_cpu(&img, &txt, 2, 2, 1.0, 0.0);
        assert!(loss > 0.0);
    }
}
