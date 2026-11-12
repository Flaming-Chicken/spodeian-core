//! Rotation and Incoherence PTQ: QuaRot, SpinQuant, and QuIP#
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Device, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn hadamard_recursive(dim: usize) -> Vec<f32> {
    if dim == 1 {
        return vec![1.0f32];
    }
    let half = dim / 2;
    let sub = hadamard_recursive(half);
    let factor = 1.0f32 / (2.0f32.sqrt());
    let mut out = vec![0.0f32; dim * dim];

    for r in 0..half {
        for c in 0..half {
            let v = sub[r * half + c] * factor;
            out[r * dim + c] = v;
            out[r * dim + (c + half)] = v;
            out[(r + half) * dim + c] = v;
            out[(r + half) * dim + (c + half)] = -v;
        }
    }
    out
}

/// Recursively constructs a normalized Walsh-Hadamard matrix of size dim = 2^k.
pub fn construct_hadamard_matrix(dim: usize, device: &Device) -> Result<Tensor> {
    let data = hadamard_recursive(dim);
    Tensor::from_vec(data, (dim, dim), device)
}

/// Constructs orthogonal randomized Hadamard matrix: Q = H_dim @ diag(signs).
pub fn construct_randomized_hadamard(dim: usize, signs: &Tensor) -> Result<Tensor> {
    let h = construct_hadamard_matrix(dim, signs.device())?;
    let s_vec = signs.to_vec1::<f32>()?;
    let h_vec = h.to_vec2::<f32>()?;
    let mut out_flat = vec![0.0f32; dim * dim];

    for r in 0..dim {
        for c in 0..dim {
            out_flat[r * dim + c] = h_vec[r][c] * s_vec[c];
        }
    }
    Tensor::from_vec(out_flat, (dim, dim), signs.device())
}

/// Gauss-Jordan matrix inversion with partial pivoting.
fn invert_matrix(mat: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = a[i * n + i].abs();
        for k in (i + 1)..n {
            let val = a[k * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }
        if max_val < 1e-12 {
            candle_core::bail!("Singular matrix in Cayley transform");
        }
        if pivot_row != i {
            for j in 0..n {
                a.swap(i * n + j, pivot_row * n + j);
                inv.swap(i * n + j, pivot_row * n + j);
            }
        }
        let diag = a[i * n + i];
        for j in 0..n {
            a[i * n + j] /= diag;
            inv[i * n + j] /= diag;
        }
        for k in 0..n {
            if k != i {
                let factor = a[k * n + i];
                for j in 0..n {
                    a[k * n + j] -= factor * a[i * n + j];
                    inv[k * n + j] -= factor * inv[i * n + j];
                }
            }
        }
    }
    Ok(inv)
}

/// Computes orthogonal Cayley transform: R = (I - A)(I + A)^{-1}.
pub fn cayley_orthogonal_rotation(skew_matrix: &Tensor) -> Result<Tensor> {
    let (dim, _) = skew_matrix.dims2()?;
    let s_vec = skew_matrix.to_vec2::<f32>()?;
    let mut eye_plus_a = vec![0.0f32; dim * dim];
    let mut eye_minus_a = vec![0.0f32; dim * dim];

    for i in 0..dim {
        for j in 0..dim {
            let a_ij = 0.5 * (s_vec[i][j] - s_vec[j][i]);
            let delta = if i == j { 1.0 } else { 0.0 };
            eye_plus_a[i * dim + j] = delta + a_ij;
            eye_minus_a[i * dim + j] = delta - a_ij;
        }
    }

    let inv_plus = invert_matrix(&eye_plus_a, dim)?;
    let inv_tensor = Tensor::from_vec(inv_plus, (dim, dim), skew_matrix.device())?;
    let minus_tensor = Tensor::from_vec(eye_minus_a, (dim, dim), skew_matrix.device())?;
    inv_tensor.matmul(&minus_tensor)
}

/// Symmetric per-token INT4 quantization [-8, 7].
pub fn quantize_per_token_int4(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, d) = x.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; b * d];
    let mut scales = vec![0.0f32; b];

    for i in 0..b {
        let mut max_abs = 0.0f32;
        for j in 0..d {
            let v = x_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        scales[i] = s;
        for j in 0..d {
            let orig = x_vec[i][j];
            let q = (orig / s).round().clamp(-8.0, 7.0);
            deq_vec[i * d + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (b, d), x.device())?;
    let scale_tensor = Tensor::from_vec(scales, (b, 1), x.device())?;
    Ok((dequant, scale_tensor))
}

/// Symmetric per-column INT4 quantization [-8, 7].
pub fn quantize_per_column_int4(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = w.dims2()?;
    let w_vec = w.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; d_in * d_out];
    let mut scales = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let v = w_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
        scales[j] = s;
        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / s).round().clamp(-8.0, 7.0);
            deq_vec[i * d_out + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (d_in, d_out), w.device())?;
    let scale_tensor = Tensor::from_vec(scales, (1, d_out), w.device())?;
    Ok((dequant, scale_tensor))
}

/// Simulated W4A4 GEMM with per-token activation and per-column weight quantization.
pub fn w4a4_gemm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (deq_x, _) = quantize_per_token_int4(x)?;
    let (deq_w, _) = quantize_per_column_int4(weight)?;
    deq_x.matmul(&deq_w)
}
