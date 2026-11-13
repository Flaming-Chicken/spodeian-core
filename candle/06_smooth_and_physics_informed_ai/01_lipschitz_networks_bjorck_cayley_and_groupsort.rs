//! Lipschitz Networks: Björck Orthonormalization, Cayley Transform, and GroupSort
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Iterative Björck orthonormalization: W_{k+1} = W_k (1.5 * I - 0.5 * W_k^T W_k).
pub fn bjorck_orthonormalize(w: &Tensor, iters: usize) -> Result<Tensor> {
    let norm = (w.sqr()?.sum_all()?.to_scalar::<f32>()? as f64).sqrt() + 1e-6;
    let mut w_k = (w * (1.0 / norm))?;
    let d_out = w.dims()[1];
    let eye = Tensor::eye(d_out, DType::F32, w.device())?;

    for _ in range_0_to(iters) {
        let gram = w_k.t()?.matmul(&w_k)?;
        let eye_scaled = (&eye * 1.5)?;
        let gram_scaled = (&gram * 0.5)?;
        let factor = (eye_scaled - gram_scaled)?;
        w_k = w_k.matmul(&factor)?;
    }
    Ok(w_k)
}

fn range_0_to(n: usize) -> impl Iterator<Item = usize> {
    0..n
}

fn invert_matrix(mat: &[f32], n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * (2 * n) + j] = mat[i * n + j];
        }
        a[i * (2 * n) + n + i] = 1.0;
    }
    for i in 0..n {
        let mut pivot = i;
        for k in (i + 1)..n {
            if a[k * (2 * n) + i].abs() > a[pivot * (2 * n) + i].abs() {
                pivot = k;
            }
        }
        if pivot != i {
            for j in 0..(2 * n) {
                a.swap(i * (2 * n) + j, pivot * (2 * n) + j);
            }
        }
        let diag = a[i * (2 * n) + i];
        for j in 0..(2 * n) {
            a[i * (2 * n) + j] /= diag;
        }
        for k in 0..n {
            if k != i {
                let factor = a[k * (2 * n) + i];
                for j in 0..(2 * n) {
                    a[k * (2 * n) + j] -= factor * a[i * (2 * n) + j];
                }
            }
        }
    }
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * (2 * n) + n + j];
        }
    }
    inv
}

/// Cayley transform: Q = (I - A)(I + A)^(-1) where A = M - M^T.
pub fn cayley_transform(m: &Tensor) -> Result<Tensor> {
    let d = m.dims()[0];
    let eye = Tensor::eye(d, DType::F32, m.device())?;
    let a = (m - m.t()?)?;
    let i_minus_a = (&eye - &a)?;
    let i_plus_a = (&eye + &a)?;

    let i_plus_a_vec = i_plus_a
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let inv_vec = invert_matrix(&i_plus_a_vec, d);
    let inv_tensor = Tensor::from_vec(inv_vec, (d, d), m.device())?;
    i_minus_a.matmul(&inv_tensor)
}

/// GroupSort-2 (PairSort) activation function: sorts adjacent pairs in descending order.
pub fn groupsort2(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let b = dims[0];
    let d = dims[1];
    let pairs = x.reshape((b, d / 2, 2))?;
    let a = pairs.narrow(2, 0, 1)?;
    let b_elem = pairs.narrow(2, 1, 1)?;
    let max_elem = a.maximum(&b_elem)?;
    let min_elem = a.minimum(&b_elem)?;
    let sorted = Tensor::cat(&[&max_elem, &min_elem], 2)?;
    sorted.reshape((b, d))
}
