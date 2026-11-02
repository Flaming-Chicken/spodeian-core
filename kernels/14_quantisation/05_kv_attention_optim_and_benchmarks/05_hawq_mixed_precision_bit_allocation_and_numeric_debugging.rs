//! CPU and WGSL kernels for HAWQ Mixed Precision Bit Allocation and Numeric Debugging.

/// WGSL compute shader for Hutchinson Hessian trace randomized probe estimation: `||x * z||^2`.
pub const HAWQ_HUTCHINSON_MATVEC_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> probe_z: array<f32>;
@group(0) @binding(3) var<storage, read_write> quad_out: array<f32>;

var<workgroup> s_norm_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn hutchinson_probe_kernel(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let b = wid.x;
    if (b >= u.batch) { return; }

    let tid = lid.x;
    var dot_val: f32 = 0.0;

    for (var i = tid; i < u.d_in; i += 256u) {
        dot_val += activations[b * u.d_in + i] * probe_z[i];
    }
    s_norm_sum[tid] = dot_val;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_norm_sum[tid] += s_norm_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let full_dot = s_norm_sum[0];
        quad_out[b] = full_dot * full_dot;
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

/// CPU symmetric per-row uniform weight quantization for specified bits.
pub fn quantize_weight_bits_cpu(w: &[f32], rows: usize, cols: usize, bits: usize) -> Vec<f32> {
    let mut w_hat = vec![0.0f32; rows * cols];
    let levels = ((1usize << (bits - 1)) - 1) as f32;

    for r in 0..rows {
        let row_slice = &w[r * cols..(r + 1) * cols];
        let max_abs = row_slice.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / levels;
        let scale_eps = scale + 1e-8;

        for c in 0..cols {
            let val = row_slice[c];
            let q = (val / scale_eps).round().clamp(-levels, levels);
            w_hat[r * cols + c] = q * scale;
        }
    }
    w_hat
}

/// CPU Hutchinson trace estimator for linear layer Hessian: Tr(H) = E[||x * z||^2].
pub fn hutchinson_trace_linear_cpu(
    x: &[f32],
    batch: usize,
    d_in: usize,
    probes: &[f32],
    num_probes: usize,
) -> f32 {
    let mut total_quad = 0.0f32;
    for p in 0..num_probes {
        let z = &probes[p * d_in..(p + 1) * d_in];
        let mut quad_sum = 0.0f32;
        for b in 0..batch {
            let mut dot = 0.0f32;
            for i in 0..d_in {
                dot += x[b * d_in + i] * z[i];
            }
            quad_sum += dot * dot;
        }
        total_quad += quad_sum;
    }
    total_quad / (num_probes as f32 * batch as f32)
}

/// Evaluates layerwise Signal-to-Quantization-Noise Ratio (SQNR) in dB.
pub fn layerwise_sqnr_cpu(fp_act: &[f32], quant_act: &[f32]) -> f32 {
    let mut sig_sum = 0.0f32;
    let mut noise_sum = 0.0f32;
    for (&a, &a_q) in fp_act.iter().zip(quant_act.iter()) {
        sig_sum += a * a;
        let diff = a - a_q;
        noise_sum += diff * diff;
    }
    let sig = sig_sum / fp_act.len() as f32;
    let noise = noise_sum / fp_act.len() as f32;
    10.0 * (sig / (noise + 1e-12)).log10()
}

/// Auto-dispatching Hutchinson trace estimator.
pub fn hutchinson_trace_auto(
    x: &[f32],
    batch: usize,
    d_in: usize,
    probes: &[f32],
    num_probes: usize,
) -> f32 {
    hutchinson_trace_linear_cpu(x, batch, d_in, probes, num_probes)
}
