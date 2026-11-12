//! CPU and WGSL kernels for Quantized Gradients QSGD, TernGrad, and 1-bit Adam with Error Feedback.

/// WGSL compute shader for fused 1-bit Adam Error Feedback momentum compression.
pub const ONEBIT_ADAM_EF_COMPRESS_WGSL: &str = r#"
struct Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> momentum: array<f32>;
@group(0) @binding(2) var<storage, read> error_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> m_hat_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> error_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> scale_out: array<f32>;

var<workgroup> s_abs_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn onebit_adam_compress_kernel(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x;
    if (row >= u.rows) { return; }

    let tid = lid.x;
    var local_sum: f32 = 0.0;

    for (var col = tid; col < u.cols; col += 256u) {
        let idx = row * u.cols + col;
        let m_tilde = momentum[idx] + error_in[idx];
        local_sum += abs(m_tilde);
    }
    s_abs_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_abs_sum[tid] += s_abs_sum[tid + s];
        }
        workgroupBarrier();
    }

    let row_scale = s_abs_sum[0] / f32(u.cols);
    if (tid == 0u) {
        scale_out[row] = row_scale;
    }
    workgroupBarrier();

    for (var col = tid; col < u.cols; col += 256u) {
        let idx = row * u.cols + col;
        let m_tilde = momentum[idx] + error_in[idx];
        let sign_val: f32 = select(-1.0, 1.0, m_tilde >= 0.0);
        let m_hat = row_scale * sign_val;
        m_hat_out[idx] = m_hat;
        error_out[idx] = m_tilde - m_hat;
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

/// CPU implementation of QSGD quantization with L2 norm scaling.
pub fn qsgd_quantize_cpu(
    grad: &[f32],
    rows: usize,
    cols: usize,
    num_levels: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut quant_grad = vec![0.0f32; rows * cols];
    let mut norms = vec![0.0f32; rows];
    let s = num_levels as f32;

    for r in 0..rows {
        let row_slice = &grad[r * cols..(r + 1) * cols];
        let norm = row_slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
        norms[r] = norm;
        let norm_eps = norm + 1e-8;

        for c in 0..cols {
            let g = row_slice[c];
            let xi = g.abs() / norm_eps;
            let tilde_xi = (s * xi).round() / s;
            let sign = if g > 0.0 {
                1.0
            } else if g < 0.0 {
                -1.0
            } else {
                0.0
            };
            quant_grad[r * cols + c] = sign * norm * tilde_xi;
        }
    }
    (quant_grad, norms)
}

/// CPU implementation of TernGrad quantization with outlier clipping.
pub fn terngrad_quantize_cpu(
    grad: &[f32],
    rows: usize,
    cols: usize,
    clip_factor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut quant_grad = vec![0.0f32; rows * cols];
    let mut scales = vec![0.0f32; rows];

    for r in 0..rows {
        let row_slice = &grad[r * cols..(r + 1) * cols];
        let mean = row_slice.iter().sum::<f32>() / cols as f32;
        let var = row_slice
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / cols as f32;
        let std = (var + 1e-8).sqrt();
        let c_bound = clip_factor * std;

        let mut max_abs = 0.0f32;
        let mut clipped = vec![0.0f32; cols];
        for c in 0..cols {
            let val = row_slice[c].clamp(-c_bound, c_bound);
            clipped[c] = val;
            let abs_val = val.abs();
            if abs_val > max_abs {
                max_abs = abs_val;
            }
        }
        scales[r] = max_abs;
        let max_eps = max_abs + 1e-8;

        for c in 0..cols {
            let g = clipped[c];
            let norm_g = g.abs() / max_eps;
            let mask = if norm_g >= 0.5 { 1.0 } else { 0.0 };
            let sign = if g > 0.0 {
                1.0
            } else if g < 0.0 {
                -1.0
            } else {
                0.0
            };
            quant_grad[r * cols + c] = sign * mask * max_abs;
        }
    }
    (quant_grad, scales)
}

/// CPU reference for 1-bit Adam with Error Feedback compression.
pub fn onebit_adam_compress_cpu(
    momentum: &[f32],
    error_buffer: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut m_hat = vec![0.0f32; rows * cols];
    let mut error_out = vec![0.0f32; rows * cols];
    let mut scales = vec![0.0f32; rows];

    for r in 0..rows {
        let mut abs_sum = 0.0f32;
        for c in 0..cols {
            let idx = r * cols + c;
            abs_sum += (momentum[idx] + error_buffer[idx]).abs();
        }
        let scale = abs_sum / cols as f32;
        scales[r] = scale;

        for c in 0..cols {
            let idx = r * cols + c;
            let m_tilde = momentum[idx] + error_buffer[idx];
            let sign = if m_tilde >= 0.0 { 1.0 } else { -1.0 };
            let val = scale * sign;
            m_hat[idx] = val;
            error_out[idx] = m_tilde - val;
        }
    }
    (m_hat, error_out, scales)
}

/// Auto-dispatching 1-bit Adam compression with fallback to CPU.
pub fn onebit_adam_compress_auto(
    momentum: &[f32],
    error_buffer: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    onebit_adam_compress_cpu(momentum, error_buffer, rows, cols)
}
