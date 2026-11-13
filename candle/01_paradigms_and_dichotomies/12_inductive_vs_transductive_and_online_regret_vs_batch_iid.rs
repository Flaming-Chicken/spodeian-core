//! Candle implementation for Category 01 Topic 12:
//! Inductive vs. Transductive Learning and Online Regret vs. Batch IID.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

/// Inverts an n x n float matrix via Gauss-Jordan elimination with partial pivoting.
fn invert_square_matrix(mat: &[f32], n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = mat[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_pivot = a[i * 2 * n + i].abs();
        for r in (i + 1)..n {
            let val = a[r * 2 * n + i].abs();
            if val > max_pivot {
                max_pivot = val;
                pivot_row = r;
            }
        }
        if pivot_row != i {
            for c in 0..(2 * n) {
                a.swap(i * 2 * n + c, pivot_row * 2 * n + c);
            }
        }

        let inv_p = 1.0 / a[i * 2 * n + i];
        for c in 0..(2 * n) {
            a[i * 2 * n + c] *= inv_p;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r * 2 * n + i];
                for c in 0..(2 * n) {
                    a[r * 2 * n + c] -= factor * a[i * 2 * n + c];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    inv
}

/// Transductive graph label spreading: F* = (I - alpha * S)^{-1} Y.
pub fn transductive_label_spreading(
    laplacian_s: &Tensor,
    y_init: &Tensor,
    alpha: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let num_nodes = laplacian_s.dims()[0];
    let device = laplacian_s.device();

    let s_flat = laplacian_s
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let mut m = vec![0.0f32; num_nodes * num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            let eye_ij = if i == j { 1.0f32 } else { 0.0f32 };
            m[i * num_nodes + j] = eye_ij - (alpha as f32) * s_flat[i * num_nodes + j];
        }
    }

    let inv_m = invert_square_matrix(&m, num_nodes);
    let inv_tensor = Tensor::from_vec(inv_m, (num_nodes, num_nodes), device)?;
    let f_star = inv_tensor.matmul(y_init)?;
    let f_probs = candle_nn::ops::softmax(&f_star, 1)?;
    let preds = f_probs.argmax(1)?.to_dtype(DType::F32)?;

    Ok((f_star, f_probs, preds))
}

/// Inductive linear predictor on out-of-sample points.
pub fn inductive_linear_predictor(x_test: &Tensor, w_ind: &Tensor) -> Result<(Tensor, Tensor)> {
    let dim = w_ind.dims()[0];
    let w_col = w_ind.reshape((dim, 1))?;
    let scores = x_test.matmul(&w_col)?.reshape(x_test.dims()[0])?;
    let preds = scores.ge(0.5)?.to_dtype(DType::F32)?;
    Ok((scores, preds))
}

/// FTRL-Proximal online coordinate descent updates with exact L1 sparsity.
pub fn ftrl_proximal_step(
    grad_stream: &Tensor,
    dim: usize,
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = grad_stream.device();
    let grads = grad_stream.to_vec2::<f32>()?;
    let time_steps = grads.len();

    let mut z = vec![0.0f32; dim];
    let mut n = vec![0.0f32; dim];

    for t in 0..time_steps {
        let g = &grads[t];
        let mut w_t = vec![0.0f32; dim];
        for i in 0..dim {
            if z[i].abs() <= lambda1 {
                w_t[i] = 0.0;
            } else {
                let sign_z = if z[i] > 0.0 { 1.0f32 } else { -1.0f32 };
                let denom = (beta + n[i].sqrt()) / alpha + lambda2;
                w_t[i] = -(z[i] - sign_z * lambda1) / denom;
            }
        }

        for i in 0..dim {
            let sigma_i = ((n[i] + g[i] * g[i]).sqrt() - n[i].sqrt()) / alpha;
            z[i] += g[i] - sigma_i * w_t[i];
            n[i] += g[i] * g[i];
        }
    }

    let mut final_w = vec![0.0f32; dim];
    for i in 0..dim {
        if z[i].abs() <= lambda1 {
            final_w[i] = 0.0;
        } else {
            let sign_z = if z[i] > 0.0 { 1.0f32 } else { -1.0f32 };
            let denom = (beta + n[i].sqrt()) / alpha + lambda2;
            final_w[i] = -(z[i] - sign_z * lambda1) / denom;
        }
    }

    let w_tensor = Tensor::from_vec(final_w, (dim,), device)?;
    let z_tensor = Tensor::from_vec(z, (dim,), device)?;
    let n_tensor = Tensor::from_vec(n, (dim,), device)?;

    Ok((w_tensor, z_tensor, n_tensor))
}

/// Page-Hinkley cumulative sum (CUSUM) test for streaming concept drift.
pub fn page_hinkley_cusum(
    stream_errors: &Tensor,
    delta: f32,
    threshold: f32,
    mean_est: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = stream_errors.device();
    let errors = stream_errors.to_vec1::<f32>()?;
    let time_steps = errors.len();

    let mut cusum_s = Vec::with_capacity(time_steps);
    let mut ph_stat = Vec::with_capacity(time_steps);
    let mut drift_flags = Vec::with_capacity(time_steps);

    let mut cum_s = 0.0f32;
    let mut min_s = 0.0f32;

    for t in 0..time_steps {
        cum_s += errors[t] - mean_est - delta;
        if cum_s < min_s {
            min_s = cum_s;
        }
        let stat = cum_s - min_s;
        cusum_s.push(cum_s);
        ph_stat.push(stat);
        drift_flags.push(if stat > threshold { 1.0f32 } else { 0.0f32 });
    }

    let s_tensor = Tensor::from_vec(cusum_s, (time_steps,), device)?;
    let ph_tensor = Tensor::from_vec(ph_stat, (time_steps,), device)?;
    let d_tensor = Tensor::from_vec(drift_flags, (time_steps,), device)?;

    Ok((s_tensor, ph_tensor, d_tensor))
}

/// Top-level module encapsulating linear anchor.
pub struct InductiveTransductiveModule {
    pub linear: Linear,
}

impl InductiveTransductiveModule {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self {
            linear: Linear::new(weight, Some(bias)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}
