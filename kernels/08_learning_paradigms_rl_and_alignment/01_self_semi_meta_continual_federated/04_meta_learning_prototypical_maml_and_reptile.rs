//! CPU and WGSL kernels for Meta Learning Prototypical MAML and Reptile.

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

/// Fused ProtoNet negative squared Euclidean distance CPU kernel:
/// Given queries Q x D and prototypes N x D, computes Q x N negative distances:
/// logits[q, n] = - sum_d (queries[q, d] - prototypes[n, d])^2.
pub fn protonet_euclidean_dist_cpu(
    queries: &[f32],
    prototypes: &[f32],
    q: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; q * n];
    for qi in 0..q {
        let q_offset = qi * d;
        for ni in 0..n {
            let n_offset = ni * d;
            let mut sum_sq = 0.0f32;
            for di in 0..d {
                let diff = queries[q_offset + di] - prototypes[n_offset + di];
                sum_sq += diff * diff;
            }
            logits[qi * n + ni] = -sum_sq;
        }
    }
    logits
}

/// Fused ProtoNet negative squared Euclidean distance WGSL compute shader
pub const PROTONET_EUCLIDEAN_DIST_WGSL: &str = r#"
struct Config {
    q: u32,
    n: u32,
    d: u32,
};

@group(0) @binding(0) var<storage, read> queries: array<f32>;
@group(0) @binding(1) var<storage, read> prototypes: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ni = global_id.x;
    let qi = global_id.y;

    if (qi >= cfg.q || ni >= cfg.n) {
        return;
    }

    let q_offset = qi * cfg.d;
    let n_offset = ni * cfg.d;
    var sum_sq: f32 = 0.0;

    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        let diff = queries[q_offset + di] - prototypes[n_offset + di];
        sum_sq = sum_sq + diff * diff;
    }

    logits[qi * cfg.n + ni] = -sum_sq;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn protonet_euclidean_dist_auto(
    queries: &[f32],
    prototypes: &[f32],
    q: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    protonet_euclidean_dist_cpu(queries, prototypes, q, n, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protonet_euclidean_dist_cpu() {
        let q = vec![1.0, 2.0]; // [1, 2]
        let p = vec![1.0, 2.0, 0.0, 0.0]; // [2, 2]
        let dists = protonet_euclidean_dist_cpu(&q, &p, 1, 2, 2);
        assert!((dists[0] - 0.0).abs() < 1e-5);
        assert!((dists[1] - (-5.0)).abs() < 1e-5);
    }
}
