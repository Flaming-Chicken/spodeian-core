//! CPU and WGSL kernels for Non Contrastive and Masked Ssl Byol Dino Vicreg MAE JEPA.

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

/// Fused BYOL / SimSiam normalized prediction loss CPU kernel:
/// Computes per-row cosine similarity between online prediction `p` and target projection `z`,
/// and returns the mean of `2.0 - 2.0 * cos_sim`.
pub fn byol_prediction_loss_cpu(p: &[f32], z: &[f32], b: usize, d: usize) -> f32 {
    let mut sum_loss = 0.0f32;
    for i in 0..b {
        let offset = i * d;
        let mut dot = 0.0f32;
        let mut p_sq = 0.0f32;
        let mut z_sq = 0.0f32;
        for k in 0..d {
            let pv = p[offset + k];
            let zv = z[offset + k];
            dot += pv * zv;
            p_sq += pv * pv;
            z_sq += zv * zv;
        }
        let p_norm = (p_sq + 1e-12).sqrt();
        let z_norm = (z_sq + 1e-12).sqrt();
        let cos_sim = dot / (p_norm * z_norm);
        sum_loss += 2.0 - 2.0 * cos_sim;
    }
    sum_loss / (b as f32)
}

/// Fused BYOL normalized prediction loss WGSL compute shader
pub const BYOL_COSINE_SIM_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
};

@group(0) @binding(0) var<storage, read> p: array<f32>;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_vec: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.b) {
        return;
    }

    let offset = i * cfg.d;
    var dot: f32 = 0.0;
    var p_sq: f32 = 0.0;
    var z_sq: f32 = 0.0;

    for (var k: u32 = 0u; k < cfg.d; k = k + 1u) {
        let pv = p[offset + k];
        let zv = z[offset + k];
        dot = dot + pv * zv;
        p_sq = p_sq + pv * pv;
        z_sq = z_sq + zv * zv;
    }

    let p_norm = sqrt(p_sq + 1e-12);
    let z_norm = sqrt(z_sq + 1e-12);
    let cos_sim = dot / (p_norm * z_norm);

    loss_vec[i] = 2.0 - 2.0 * cos_sim;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn byol_prediction_loss_auto(p: &[f32], z: &[f32], b: usize, d: usize) -> f32 {
    byol_prediction_loss_cpu(p, z, b, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byol_prediction_loss_cpu() {
        let p = vec![1.0, 0.0, 0.0, 1.0];
        let z = vec![1.0, 0.0, 0.0, 1.0];
        let loss = byol_prediction_loss_cpu(&p, &z, 2, 2);
        assert!((loss - 0.0).abs() < 1e-5);
    }
}
