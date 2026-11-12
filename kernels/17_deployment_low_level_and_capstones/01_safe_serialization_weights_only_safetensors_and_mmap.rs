//! CPU and WGSL kernels for Safe Serialization, Safetensors Wire Format, and Memory Mapping.

/// WGSL compute shader for fused mmap buffer affine transformation upon initial device upload.
pub const FUSED_MMAP_AFFINE_WGSL: &str = r#"
struct Uniforms {
    size: u32,
    scale: f32,
    bias: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> raw_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn mmap_affine_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.size) { return; }
    out[idx] = raw_in[idx] * u.scale + u.bias;
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

/// Zero-copy reinterpret byte slice to f32 slice (mmap analogue).
pub fn mmap_bytes_to_f32_slice(bytes: &[u8]) -> &[f32] {
    let ptr = bytes.as_ptr() as *const f32;
    let len = bytes.len() / std::mem::size_of::<f32>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// CPU affine copy for mmapped weights: `out[i] = raw[i] * scale + bias`.
pub fn mmap_affine_copy_cpu(raw_in: &[f32], scale: f32, bias: f32) -> Vec<f32> {
    raw_in.iter().map(|&x| x * scale + bias).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_bytes_to_f32_slice() {
        let floats = vec![1.5f32, -2.5f32, 3.0f32, 4.25f32];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                floats.as_ptr() as *const u8,
                floats.len() * std::mem::size_of::<f32>(),
            )
        };
        let reinterpreted = mmap_bytes_to_f32_slice(bytes);
        assert_eq!(reinterpreted, floats.as_slice());
    }

    #[test]
    fn test_mmap_affine_copy_cpu() {
        let raw = vec![1.0f32, 2.0f32, 3.0f32];
        let out = mmap_affine_copy_cpu(&raw, 2.0, 1.0);
        assert_eq!(out, vec![3.0, 5.0, 7.0]);
    }
}
