//! Low Rank Decomposition SVD Asvd Tucker Monarch and Tt
//! in candle/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Truncated SVD rank-r reconstruction using power iteration with Hotelling deflation.
pub fn truncated_svd(w: &Tensor, rank: usize) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let mut w_curr = w.flatten_all()?.to_vec1::<f32>()?;
    let mut w_recon = vec![0.0f32; rows * cols];
    let mut sigmas = Vec::with_capacity(rank);

    for _ in 0..rank {
        // Initial normalized probe vector
        let mut v = vec![1.0f32 / (cols as f32).sqrt(); cols];
        for _ in 0..70 {
            // u = W_curr * v
            let mut u = vec![0.0f32; rows];
            for r in 0..rows {
                let mut sum = 0.0f32;
                for c in 0..cols {
                    sum += w_curr[r * cols + c] * v[c];
                }
                u[r] = sum;
            }
            let u_norm = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for r in 0..rows {
                u[r] /= u_norm;
            }

            // v = W_curr^T * u
            let mut v_next = vec![0.0f32; cols];
            for c in 0..cols {
                let mut sum = 0.0f32;
                for r in 0..rows {
                    sum += w_curr[r * cols + c] * u[r];
                }
                v_next[c] = sum;
            }
            let v_norm = v_next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for c in 0..cols {
                v[c] = v_next[c] / v_norm;
            }
        }

        // Final singular triplet evaluation
        let mut u = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += w_curr[r * cols + c] * v[c];
            }
            u[r] = sum;
        }
        let sigma = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sigma > 1e-12 {
            for r in 0..rows {
                u[r] /= sigma;
            }
        }
        sigmas.push(sigma);

        // Accumulate into w_recon and deflate w_curr
        for r in 0..rows {
            for c in 0..cols {
                let delta = sigma * u[r] * v[c];
                w_recon[r * cols + c] += delta;
                w_curr[r * cols + c] -= delta;
            }
        }
    }

    let recon_tensor = Tensor::from_vec(w_recon, (rows, cols), w.device())?;
    let sigmas_tensor = Tensor::from_vec(sigmas, (rank,), w.device())?;
    Ok((recon_tensor, sigmas_tensor))
}

/// Activation-aware SVD (ASVD): scales columns by activation variance before SVD.
pub fn asvd_decompose(w: &Tensor, act_scales: &Tensor, rank: usize) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let s_vec = act_scales.flatten_all()?.to_vec1::<f32>()?;

    let mut w_scaled = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = s_vec[c].max(1e-5);
            w_scaled[r * cols + c] = w_vec[r * cols + c] * s;
        }
    }

    let w_scaled_tensor = Tensor::from_vec(w_scaled, (rows, cols), w.device())?;
    let (w_scaled_recon, sigmas) = truncated_svd(&w_scaled_tensor, rank)?;
    let recon_vec = w_scaled_recon.flatten_all()?.to_vec1::<f32>()?;

    let mut w_asvd = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = s_vec[c].max(1e-5);
            w_asvd[r * cols + c] = recon_vec[r * cols + c] / s;
        }
    }

    let asvd_tensor = Tensor::from_vec(w_asvd, (rows, cols), w.device())?;
    Ok((asvd_tensor, sigmas))
}

/// Monarch factorized forward transformation using two block-diagonal stages.
pub fn monarch_forward(x: &Tensor, b1: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let (batch, dim) = x.dims2()?;
    if dim != 16 {
        candle_core::bail!("Expected dim = 16, got {dim}");
    }
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let b1_vec = b1.flatten_all()?.to_vec1::<f32>()?; // [4, 4, 4]
    let b2_vec = b2.flatten_all()?.to_vec1::<f32>()?; // [4, 4, 4]

    let mut y_vec = vec![0.0f32; batch * 16];

    for b in 0..batch {
        // Stage 1: x[b, k, i] * b1[k, i, j] -> h1[k, j]
        let mut h1 = [[0.0f32; 4]; 4];
        for k in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for i in 0..4 {
                    let x_val = x_vec[b * 16 + k * 4 + i];
                    let b1_val = b1_vec[k * 16 + i * 4 + j];
                    sum += x_val * b1_val;
                }
                h1[k][j] = sum;
            }
        }

        // Stage 2: Transpose block axes: h1_perm[k, j] = h1[j][k]
        let mut h1_perm = [[0.0f32; 4]; 4];
        for k in 0..4 {
            for j in 0..4 {
                h1_perm[k][j] = h1[j][k];
            }
        }

        // Stage 3: h1_perm[k, i] * b2[k, i, j] -> h2[k, j]
        for k in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for i in 0..4 {
                    let h_val = h1_perm[k][i];
                    let b2_val = b2_vec[k * 16 + i * 4 + j];
                    sum += h_val * b2_val;
                }
                y_vec[b * 16 + k * 4 + j] = sum;
            }
        }
    }

    Tensor::from_vec(y_vec, (batch, 16), x.device())
}
