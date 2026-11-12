//! RTN vs GPTQ Hessian Error Compensation
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Standard Round-To-Nearest (RTN) per-column symmetric quantization.
pub fn rtn_quantize(weight: &Tensor, qmin: f32, qmax: f32) -> Result<Tensor> {
    let (d_in, d_out) = weight.dims2()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let mut out_vec = vec![0.0f32; d_in * d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let val = w_vec[i][j].abs();
            if val > max_abs {
                max_abs = val;
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / qmax };
        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / scale).round().clamp(qmin, qmax) * scale;
            out_vec[i * d_out + j] = q;
        }
    }

    Tensor::from_vec(out_vec, (d_in, d_out), weight.device())
}

/// Gauss-Jordan matrix inversion with partial pivoting for an n x n row-major matrix.
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
            candle_core::bail!("Matrix is singular or near-singular in Gauss-Jordan inversion");
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

/// Computes empirical activation Hessian H = 2 * X^T X / N and damped inverse.
pub fn compute_damped_hessian_inv(x: &Tensor, damp_factor: f64) -> Result<(Tensor, Tensor)> {
    let (n_samples, d_in) = x.dims2()?;
    let xt = x.t()?;
    let factor = 2.0 / (n_samples as f64);
    let h = (xt.matmul(x)? * factor)?;

    let h_vec = h.to_vec2::<f32>()?;
    let mut diag_sum = 0.0f64;
    for i in 0..d_in {
        diag_sum += h_vec[i][i] as f64;
    }
    let diag_mean = diag_sum / (d_in as f64);
    let damp = if diag_mean > 0.0 {
        damp_factor * diag_mean
    } else {
        1e-4
    } as f32;

    let mut h_flat = vec![0.0f32; d_in * d_in];
    for i in 0..d_in {
        for j in 0..d_in {
            let mut val = h_vec[i][j];
            if i == j {
                val += damp;
            }
            h_flat[i * d_in + j] = val;
        }
    }

    let h_inv_flat = invert_matrix(&h_flat, d_in)?;
    let h_inv = Tensor::from_vec(h_inv_flat, (d_in, d_in), x.device())?;
    Ok((h, h_inv))
}

/// Second-order error compensation GPTQ column-by-column update.
pub fn gptq_quantize(weight: &Tensor, h_inv: &Tensor, qmin: f32, qmax: f32) -> Result<Tensor> {
    let (d_in, d_out) = weight.dims2()?;
    let mut w = weight.to_vec2::<f32>()?;
    let h_inv_vec = h_inv.to_vec2::<f32>()?;

    // Compute initial scales per column
    let mut scales = vec![0.0f32; d_out];
    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let val = w[i][j].abs();
            if val > max_abs {
                max_abs = val;
            }
        }
        scales[j] = if max_abs == 0.0 { 1.0 } else { max_abs / qmax };
    }

    // Step through each column/input dimension
    for c in 0..d_in {
        let h_inv_cc = h_inv_vec[c][c];
        for j in 0..d_out {
            let w_cj = w[c][j];
            let s = scales[j];
            let q_cj = (w_cj / s).round().clamp(qmin, qmax) * s;
            let err_c = w_cj - q_cj;

            if h_inv_cc != 0.0 && c + 1 < d_in {
                let step = err_c / h_inv_cc;
                for r in (c + 1)..d_in {
                    let delta = step * h_inv_vec[r][c];
                    w[r][j] -= delta;
                }
            }
            w[c][j] = q_cj;
        }
    }

    let flat: Vec<f32> = w.into_iter().flatten().collect();
    Tensor::from_vec(flat, (d_in, d_out), weight.device())
}

/// GPTQ with act-order (descending Hessian diagonal activation sorting).
pub fn gptq_with_act_order(
    x: &Tensor,
    weight: &Tensor,
    damp_factor: f64,
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (n_samples, d_in) = x.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let xt = x.t()?;
    let factor = 2.0 / (n_samples as f64);
    let h = (xt.matmul(x)? * factor)?;
    let h_vec = h.to_vec2::<f32>()?;

    let mut diag_indices: Vec<(usize, f32)> = (0..d_in).map(|i| (i, h_vec[i][i])).collect();
    diag_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let perm: Vec<usize> = diag_indices.iter().map(|(idx, _)| *idx).collect();
    let mut inv_perm = vec![0usize; d_in];
    for (new_pos, &orig_idx) in perm.iter().enumerate() {
        inv_perm[orig_idx] = new_pos;
    }

    // Permute x columns and weight rows
    let x_vec = x.to_vec2::<f32>()?;
    let mut x_perm_vec = vec![0.0f32; n_samples * d_in];
    for row in 0..n_samples {
        for col in 0..d_in {
            x_perm_vec[row * d_in + col] = x_vec[row][perm[col]];
        }
    }
    let x_perm = Tensor::from_vec(x_perm_vec, (n_samples, d_in), x.device())?;

    let w_vec = weight.to_vec2::<f32>()?;
    let mut w_perm_vec = vec![0.0f32; d_in * d_out];
    for r in 0..d_in {
        for c in 0..d_out {
            w_perm_vec[r * d_out + c] = w_vec[perm[r]][c];
        }
    }
    let w_perm = Tensor::from_vec(w_perm_vec, (d_in, d_out), weight.device())?;

    let (_, h_inv_perm) = compute_damped_hessian_inv(&x_perm, damp_factor)?;
    let w_perm_quant = gptq_quantize(&w_perm, &h_inv_perm, qmin, qmax)?;
    let w_perm_quant_vec = w_perm_quant.to_vec2::<f32>()?;

    let mut w_final_vec = vec![0.0f32; d_in * d_out];
    for r in 0..d_in {
        for c in 0..d_out {
            w_final_vec[r * d_out + c] = w_perm_quant_vec[inv_perm[r]][c];
        }
    }
    let w_final = Tensor::from_vec(w_final_vec, (d_in, d_out), weight.device())?;
    let perm_f32: Vec<f32> = perm.iter().map(|&idx| idx as f32).collect();
    let perm_tensor = Tensor::from_vec(perm_f32, (d_in,), x.device())?;

    Ok((w_final, perm_tensor, h_inv_perm))
}
