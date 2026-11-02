//! Topic 07: Input & Output Spaces - Tokens, Bytes, Patches, Coords, Pointers & Manifolds (Dual Engine: WGSL + CPU fallback).

use crate::context::{KernelDevice, WgpuContext};

pub const WGSL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
};
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= dims.batch || col >= dims.out_dim) {
        return;
    }
    var acc: f32 = b[col];
    for (var i: u32 = 0u; i < dims.in_dim; i = i + 1u) {
        acc = acc + x[row * dims.in_dim + i] * w[i * dims.out_dim + col];
    }
    y[row * dims.out_dim + col] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
}

pub fn linear_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        for j in 0..out_dim {
            let mut acc = b[j];
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[k * out_dim + j];
            }
            y[i * out_dim + j] = acc;
        }
    }
    y
}

/// Continuous 6D Gram-Schmidt projection to SO(3) 3x3 rotation matrix.
/// Returns flattened [batch, 3, 3] matrix in row-major order: [b1_x, b2_x, b3_x, b1_y, b2_y, b3_y, b1_z, b2_z, b3_z].
pub fn rotation_6d_to_matrix_cpu(d6: &[f32], batch: usize) -> Vec<f32> {
    let mut rot_matrices = vec![0.0f32; batch * 9];
    for i in 0..batch {
        let v6 = &d6[i * 6..(i + 1) * 6];
        let a1 = [v6[0], v6[1], v6[2]];
        let a2 = [v6[3], v6[4], v6[5]];

        let norm_a1 = (a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2])
            .sqrt()
            .max(1e-8);
        let b1 = [a1[0] / norm_a1, a1[1] / norm_a1, a1[2] / norm_a1];

        let dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
        let u2 = [
            a2[0] - dot * b1[0],
            a2[1] - dot * b1[1],
            a2[2] - dot * b1[2],
        ];
        let norm_u2 = (u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2])
            .sqrt()
            .max(1e-8);
        let b2 = [u2[0] / norm_u2, u2[1] / norm_u2, u2[2] / norm_u2];

        // Cross product b3 = b1 x b2
        let b3 = [
            b1[1] * b2[2] - b1[2] * b2[1],
            b1[2] * b2[0] - b1[0] * b2[2],
            b1[0] * b2[1] - b1[1] * b2[0],
        ];

        // Column vectors b1, b2, b3 form columns of 3x3 rotation matrix:
        // Row 0: [b1[0], b2[0], b3[0]]
        // Row 1: [b1[1], b2[1], b3[1]]
        // Row 2: [b1[2], b2[2], b3[2]]
        let base = i * 9;
        rot_matrices[base + 0] = b1[0];
        rot_matrices[base + 1] = b2[0];
        rot_matrices[base + 2] = b3[0];
        rot_matrices[base + 3] = b1[1];
        rot_matrices[base + 4] = b2[1];
        rot_matrices[base + 5] = b3[1];
        rot_matrices[base + 6] = b1[2];
        rot_matrices[base + 7] = b2[2];
        rot_matrices[base + 8] = b3[2];
    }
    rot_matrices
}

/// Pointer network forward pass on CPU: (ptr_logits, ptr_probs).
pub fn pointer_network_forward_cpu(
    encoder_seq: &[f32],
    decoder_state: &[f32],
    w_enc: &[f32],
    w_dec: &[f32],
    b_ptr: &[f32],
    v_ptr: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut logits = vec![0.0f32; batch * seq_len];
    let mut probs = vec![0.0f32; batch * seq_len];

    for i in 0..batch {
        let dec_i = &decoder_state[i * dim..(i + 1) * dim];
        let mut d_proj = vec![0.0f32; dim];
        for d in 0..dim {
            let mut acc = 0.0f32;
            for k in 0..dim {
                acc += dec_i[k] * w_dec[k * dim + d];
            }
            d_proj[d] = acc;
        }

        let mut max_logit = f32::NEG_INFINITY;
        for s in 0..seq_len {
            let enc_s =
                &encoder_seq[i * seq_len * dim + s * dim..i * seq_len * dim + (s + 1) * dim];
            let mut u_score = 0.0f32;
            for d in 0..dim {
                let mut e_p = 0.0f32;
                for k in 0..dim {
                    e_p += enc_s[k] * w_enc[k * dim + d];
                }
                let tanh_val = (e_p + d_proj[d] + b_ptr[d]).tanh();
                u_score += tanh_val * v_ptr[d];
            }
            logits[i * seq_len + s] = u_score;
            if u_score > max_logit {
                max_logit = u_score;
            }
        }

        let mut sum_exp = 0.0f32;
        for s in 0..seq_len {
            let exp_val = (logits[i * seq_len + s] - max_logit).exp();
            probs[i * seq_len + s] = exp_val;
            sum_exp += exp_val;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for s in 0..seq_len {
            probs[i * seq_len + s] *= inv_sum;
        }
    }

    (logits, probs)
}

/// Multi-frequency Fourier coordinate encoding on CPU.
pub fn fourier_coordinate_encode_cpu(
    coords: &[f32],
    batch: usize,
    coord_dim: usize,
    num_bands: usize,
) -> Vec<f32> {
    let out_dim = coord_dim * num_bands * 2;
    let mut out = vec![0.0f32; batch * out_dim];

    for i in 0..batch {
        let c_i = &coords[i * coord_dim..(i + 1) * coord_dim];
        let mut idx = 0;
        for d in 0..coord_dim {
            for band in 0..num_bands {
                let scale = (2.0f32.powi(band as i32)) * std::f32::consts::PI;
                let val = c_i[d] * scale;
                out[i * out_dim + idx] = val.sin();
                out[i * out_dim + idx + 1] = val.cos();
                idx += 2;
            }
        }
    }
    out
}

/// Patch linear embedding forward pass on CPU: patches @ w_patch + b_patch.
pub fn patch_embedding_forward_cpu(
    patches: &[f32],
    w_patch: &[f32],
    b_patch: &[f32],
    batch: usize,
    num_patches: usize,
    patch_dim: usize,
    embed_dim: usize,
) -> Vec<f32> {
    let total_tokens = batch * num_patches;
    linear_forward_cpu(
        patches,
        w_patch,
        b_patch,
        total_tokens,
        patch_dim,
        embed_dim,
    )
}

#[cfg(feature = "wgpu")]
pub fn linear_forward_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    let dims = Dims {
        batch: batch as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
    };

    let wg_x = ((batch as u32) + 15) / 16;
    let wg_y = ((out_dim as u32) + 15) / 16;

    let bytes = crate::wgpu_util::run_compute_shader(
        ctx,
        WGSL_SHADER,
        "main",
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(w),
            bytemuck::cast_slice(b),
        ],
        batch * out_dim * std::mem::size_of::<f32>(),
        Some(bytemuck::bytes_of(&dims)),
        (wg_x, wg_y, 1),
    )?;

    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

#[cfg(not(feature = "wgpu"))]
pub fn linear_forward_wgpu(
    _ctx: &WgpuContext,
    _x: &[f32],
    _w: &[f32],
    _b: &[f32],
    _batch: usize,
    _in_dim: usize,
    _out_dim: usize,
) -> Option<Vec<f32>> {
    None
}

pub fn linear_forward(
    device: &KernelDevice,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Wgpu(ctx) => {
            if let Some(res) = linear_forward_wgpu(ctx, x, w, b, batch, in_dim, out_dim) {
                return res;
            }
            linear_forward_cpu(x, w, b, batch, in_dim, out_dim)
        }
        KernelDevice::Cpu => linear_forward_cpu(x, w, b, batch, in_dim, out_dim),
    }
}
