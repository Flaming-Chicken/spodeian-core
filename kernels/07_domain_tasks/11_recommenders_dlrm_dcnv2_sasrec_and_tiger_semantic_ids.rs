//! CPU and WGSL kernels for Recommenders Dlrm Dcnv2 Sasrec and Tiger Semantic Ids.

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

/// Fused DCN-v2 Cross Layer kernel:
/// x_next[b, d] = x0[b, d] * (sum_k xl[b, k] * w[k, d] + bias[d]) + xl[b, d]
/// x0, xl: [B, D] row-major
/// w: [D, D] row-major
/// bias: [D]
/// out: [B, D] row-major
pub fn dcn_v2_cross_cpu(
    x0: &[f32],
    xl: &[f32],
    w: &[f32],
    bias: &[f32],
    b: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * d];
    for bi in 0..b {
        let b_offset = bi * d;
        for di in 0..d {
            let mut proj = bias[di];
            for ki in 0..d {
                proj += xl[b_offset + ki] * w[ki * d + di];
            }
            out[b_offset + di] = x0[b_offset + di] * proj + xl[b_offset + di];
        }
    }
    out
}

/// Fused DCN-v2 Cross Layer WGSL compute shader
pub const DCN_V2_CROSS_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> x0: array<f32>;
@group(0) @binding(1) var<storage, read> xl: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let di = global_id.x;
    let bi = global_id.y;

    if (bi >= cfg.b || di >= cfg.d) {
        return;
    }

    let b_offset = bi * cfg.d;
    var proj: f32 = bias[di];
    for (var ki: u32 = 0u; ki < cfg.d; ki = ki + 1u) {
        proj = proj + xl[b_offset + ki] * w[ki * cfg.d + di];
    }

    out[b_offset + di] = x0[b_offset + di] * proj + xl[b_offset + di];
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn dcn_v2_cross_auto(
    x0: &[f32],
    xl: &[f32],
    w: &[f32],
    bias: &[f32],
    b: usize,
    d: usize,
) -> Vec<f32> {
    dcn_v2_cross_cpu(x0, xl, w, bias, b, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcn_v2_cross_cpu() {
        let x0 = vec![1.0, 2.0];
        let xl = vec![0.5, 0.5];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0];
        let res = dcn_v2_cross_cpu(&x0, &xl, &w, &bias, 1, 2);
        assert_eq!(res.len(), 2);
        // proj = [0.5, 0.5]
        // out[0] = 1.0 * 0.5 + 0.5 = 1.0
        // out[1] = 2.0 * 0.5 + 0.5 = 1.5
        assert!((res[0] - 1.0).abs() < 1e-5);
        assert!((res[1] - 1.5).abs() < 1e-5);
    }
}
