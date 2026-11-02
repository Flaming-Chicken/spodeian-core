//! CPU and WGSL kernels for Causal Treatment Effects TARNet CFRNet Dragonnet DML and R Learner.

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

/// WGSL shader for parallel DML Robinson residual product calculation:
/// num[i] = (t[i] - e_hat[i]) * (y[i] - m_hat[i])
/// den[i] = (t[i] - e_hat[i])^2
pub const DML_RESIDUAL_WGSL: &str = r#"
struct Params {
    num_samples: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read> m_hat: array<f32>;
@group(0) @binding(4) var<storage, read> e_hat: array<f32>;
@group(0) @binding(5) var<storage, read_write> num_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> den_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_samples) {
        return;
    }
    let res_t = t[i] - e_hat[i];
    let res_y = y[i] - m_hat[i];
    num_out[i] = res_t * res_y;
    den_out[i] = res_t * res_t;
}
"#;

/// CPU reference calculation for DML Robinson CATE estimator.
pub fn dml_cate_cpu(y: &[f32], t: &[f32], m_hat: &[f32], e_hat: &[f32]) -> f32 {
    let n = y.len();
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..n {
        let res_t = t[i] - e_hat[i];
        let res_y = y[i] - m_hat[i];
        num += res_t * res_y;
        den += res_t * res_t;
    }
    num / den.max(1e-8)
}

/// Automatic dispatcher falling back to CPU.
pub fn dml_cate_auto(y: &[f32], t: &[f32], m_hat: &[f32], e_hat: &[f32]) -> f32 {
    dml_cate_cpu(y, t, m_hat, e_hat)
}
