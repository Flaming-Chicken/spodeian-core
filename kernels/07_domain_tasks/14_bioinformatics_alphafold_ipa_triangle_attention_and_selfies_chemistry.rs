//! CPU and WGSL kernels for Bioinformatics Alphafold Ipa Triangle Attention and Selfies Chemistry.

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

/// Fused AlphaFold IPA 3D Point Distance Penalty Kernel:
/// q_pts: [B, N, H, P, 3] row-major -> index = (((bi * N + i) * H + h) * P + p) * 3 + c
/// k_pts: [B, N, H, P, 3] row-major -> index = (((bi * N + j) * H + h) * P + p) * 3 + c
/// gamma: [H]
/// out: [B, N, N, H] row-major -> index = ((bi * N + i) * N + j) * H + h
pub fn ipa_distance_penalty_cpu(
    q_pts: &[f32],
    k_pts: &[f32],
    gamma: &[f32],
    b: usize,
    n: usize,
    h: usize,
    p: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n * n * h];
    for bi in 0..b {
        for i in 0..n {
            for j in 0..n {
                for hi in 0..h {
                    let mut sq_dist = 0.0f32;
                    for pi in 0..p {
                        let q_idx = (((bi * n + i) * h + hi) * p + pi) * 3;
                        let k_idx = (((bi * n + j) * h + hi) * p + pi) * 3;

                        let dx = q_pts[q_idx] - k_pts[k_idx];
                        let dy = q_pts[q_idx + 1] - k_pts[k_idx + 1];
                        let dz = q_pts[q_idx + 2] - k_pts[k_idx + 2];

                        sq_dist += dx * dx + dy * dy + dz * dz;
                    }
                    let out_idx = ((bi * n + i) * n + j) * h + hi;
                    out[out_idx] = -0.5 * gamma[hi] * sq_dist;
                }
            }
        }
    }
    out
}

/// Fused IPA 3D Distance Penalty WGSL compute shader
pub const IPA_DISTANCE_PENALTY_WGSL: &str = r#"
struct Config {
    b: u32,
    n: u32,
    h: u32,
    p: u32,
};

@group(0) @binding(0) var<storage, read> q_pts: array<f32>;
@group(0) @binding(1) var<storage, read> k_pts: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> penalties: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let i = global_id.y;
    let bi_hi = global_id.z;

    let bi = bi_hi / cfg.h;
    let hi = bi_hi % cfg.h;

    if (bi >= cfg.b || i >= cfg.n || j >= cfg.n || hi >= cfg.h) {
        return;
    }

    var sq_dist: f32 = 0.0;
    for (var pi: u32 = 0u; pi < cfg.p; pi = pi + 1u) {
        let q_idx = (((bi * cfg.n + i) * cfg.h + hi) * cfg.p + pi) * 3u;
        let k_idx = (((bi * cfg.n + j) * cfg.h + hi) * cfg.p + pi) * 3u;

        let dx = q_pts[q_idx] - k_pts[k_idx];
        let dy = q_pts[q_idx + 1u] - k_pts[k_idx + 1u];
        let dz = q_pts[q_idx + 2u] - k_pts[k_idx + 2u];

        sq_dist = sq_dist + (dx * dx + dy * dy + dz * dz);
    }

    let out_idx = ((bi * cfg.n + i) * cfg.n + j) * cfg.h + hi;
    penalties[out_idx] = -0.5 * gamma[hi] * sq_dist;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn ipa_distance_penalty_auto(
    q_pts: &[f32],
    k_pts: &[f32],
    gamma: &[f32],
    b: usize,
    n: usize,
    h: usize,
    p: usize,
) -> Vec<f32> {
    ipa_distance_penalty_cpu(q_pts, k_pts, gamma, b, n, h, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipa_distance_penalty_cpu() {
        let q = vec![1.0, 0.0, 0.0]; // [1, 1, 1, 1, 3]
        let k = vec![0.0, 0.0, 0.0];
        let gamma = vec![2.0];
        let res = ipa_distance_penalty_cpu(&q, &k, &gamma, 1, 1, 1, 1);
        assert_eq!(res.len(), 1);
        // sq_dist = 1.0. penalty = -0.5 * 2.0 * 1.0 = -1.0
        assert!((res[0] - (-1.0)).abs() < 1e-5);
    }
}
