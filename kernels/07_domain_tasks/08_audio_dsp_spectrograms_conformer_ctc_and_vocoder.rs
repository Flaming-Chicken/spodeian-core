//! CPU and WGSL kernels for Audio Dsp Spectrograms Conformer Ctc and Vocoder.

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

/// Computes Log-Mel Spectrogram via matrix multiplication of Mel filterbank with linear magnitude spectrum:
/// mag_spec: [B, F, T] row-major -> index = (b * F + f) * T + t
/// mel_basis: [M, F] row-major -> index = m * F + f
/// out: [B, M, T] row-major -> index = (b * M + m) * T + t
/// out[b, m, t] = ln(sum_f mel_basis[m, f] * mag_spec[b, f, t] + eps)
pub fn log_mel_filterbank_cpu(
    mag_spec: &[f32],
    mel_basis: &[f32],
    b: usize,
    f: usize,
    t: usize,
    m: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * m * t];
    for bi in 0..b {
        for mi in 0..m {
            for ti in 0..t {
                let mut sum = 0.0f32;
                for fi in 0..f {
                    let w = mel_basis[mi * f + fi];
                    let s = mag_spec[(bi * f + fi) * t + ti];
                    sum += w * s;
                }
                out[(bi * m + mi) * t + ti] = (sum + eps).ln();
            }
        }
    }
    out
}

/// Fused Log-Mel Spectrogram WGSL compute shader
pub const LOG_MEL_WGSL: &str = r#"
struct Dimensions {
    b: u32,
    f: u32,
    t: u32,
    m: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> mag_spec: array<f32>;
@group(0) @binding(1) var<storage, read> mel_basis: array<f32>;
@group(0) @binding(2) var<storage, read_write> log_mel: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let mi = global_id.x;
    let ti = global_id.y;
    let bi = global_id.z;

    if (bi >= dims.b || mi >= dims.m || ti >= dims.t) {
        return;
    }

    var sum: f32 = 0.0;
    for (var fi: u32 = 0u; fi < dims.f; fi = fi + 1u) {
        let w = mel_basis[mi * dims.f + fi];
        let s = mag_spec[(bi * dims.f + fi) * dims.t + ti];
        sum = sum + w * s;
    }

    let out_idx = (bi * dims.m + mi) * dims.t + ti;
    log_mel[out_idx] = log(sum + dims.eps);
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn log_mel_filterbank_auto(
    mag_spec: &[f32],
    mel_basis: &[f32],
    b: usize,
    f: usize,
    t: usize,
    m: usize,
    eps: f32,
) -> Vec<f32> {
    log_mel_filterbank_cpu(mag_spec, mel_basis, b, f, t, m, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_mel_cpu() {
        let b = 1;
        let f = 2;
        let t = 2;
        let m = 2;
        let mag = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let mel = vec![0.5, 0.5, 1.0, 0.0]; // [2, 2]
        let eps = 1e-5;
        let res = log_mel_filterbank_cpu(&mag, &mel, b, f, t, m, eps);
        assert_eq!(res.len(), 4);
    }
}
