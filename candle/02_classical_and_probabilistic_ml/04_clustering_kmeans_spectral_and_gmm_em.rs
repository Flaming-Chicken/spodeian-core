//! Clustering: k-Means, Spectral Clustering, and GMM EM in Native Rust Candle.
//!
//! Provides:
//! - Lloyd's k-Means assignment via pairwise distance minimization
//! - Gaussian Mixture Model log-domain Expectation-Maximization step
//! - Symmetric Normalized Graph Laplacian for Spectral Clustering: L_sym = I - D^{-1/2} A D^{-1/2}

use candle_core::{Result, Tensor};

/// Assigns each sample in X to its nearest cluster center: assignments[i] = argmin_k ||x_i - c_k||^2.
pub fn kmeans_assign(x: &Tensor, centers: &Tensor) -> Result<Tensor> {
    let (n, _d) = x.dims2()?;
    let (k, _) = centers.dims2()?;

    let x_sq = x.sqr()?.sum_keepdim(1)?; // [n, 1]
    let c_sq = centers.sqr()?.sum_keepdim(1)?.t()?; // [1, k]

    let dot = x.matmul(&centers.t()?)?; // [n, k]
    let dist_sq = ((x_sq.broadcast_as((n, k))? + c_sq.broadcast_as((n, k))?)? - (dot * 2.0)?)?;

    // Argmin across column dimension (dim 1)
    let dist_vec = dist_sq.flatten_all()?.to_vec1::<f32>()?;
    let mut assigns = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_d = f32::MAX;
        let mut min_k = 0f32;
        for c in 0..k {
            let d = dist_vec[i * k + c];
            if d < min_d {
                min_d = d;
                min_k = c as f32;
            }
        }
        assigns.push(min_k);
    }

    Tensor::from_vec(assigns, (n,), x.device())
}

/// GMM E-step: Computes posterior responsibility matrix resp [N, K] in log-domain.
pub fn gmm_e_step(x: &Tensor, mu: &Tensor, var: &Tensor, pi: &Tensor) -> Result<Tensor> {
    let (n, d) = x.dims2()?;
    let (k, _) = mu.dims2()?;

    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mu_vec = mu.flatten_all()?.to_vec1::<f32>()?;
    let var_vec = var.flatten_all()?.to_vec1::<f32>()?;
    let pi_vec = pi.flatten_all()?.to_vec1::<f32>()?;

    let log_2pi = 1.837877f32;
    let mut log_resp = vec![0.0f32; n * k];

    for i in 0..n {
        let mut max_log = f32::NEG_INFINITY;
        for c in 0..k {
            let mut quad = 0.0f32;
            let mut log_det = 0.0f32;
            for j in 0..d {
                let diff = x_vec[i * d + j] - mu_vec[c * d + j];
                let v = var_vec[c * d + j];
                quad += (diff * diff) / v;
                log_det += v.ln();
            }
            let log_p = -0.5f32 * ((d as f32) * log_2pi + log_det + quad);
            let val = (pi_vec[c] + 1e-8f32).ln() + log_p;
            log_resp[i * k + c] = val;
            if val > max_log {
                max_log = val;
            }
        }

        // Softmax normalization across K components
        let mut sum_exp = 0.0f32;
        for c in 0..k {
            sum_exp += (log_resp[i * k + c] - max_log).exp();
        }
        let log_sum = max_log + sum_exp.ln();

        for c in 0..k {
            log_resp[i * k + c] = (log_resp[i * k + c] - log_sum).exp();
        }
    }

    Tensor::from_vec(log_resp, (n, k), x.device())
}

/// Computes symmetric normalized graph Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}.
pub fn compute_symmetric_laplacian(x: &Tensor, gamma: f64) -> Result<Tensor> {
    let (n, _d) = x.dims2()?;

    let x_sq = x.sqr()?.sum_keepdim(1)?; // [n, 1]
    let dot = x.matmul(&x.t()?)?; // [n, n]
    let dist_sq = ((x_sq.broadcast_as((n, n))? + x_sq.t()?.broadcast_as((n, n))?)? - (dot * 2.0)?)?;

    let a_full = (dist_sq * (-gamma))?.exp()?;
    // Zero out diagonal
    let eye = Tensor::eye(n, candle_core::DType::F32, x.device())?;
    let ones = Tensor::ones((n, n), candle_core::DType::F32, x.device())?;
    let mask = (ones - eye)?;
    let a = a_full.mul(&mask)?;

    let degrees = a.sum_keepdim(1)?; // [n, 1]
    let clamped_d = degrees.clamp(1e-6f32, f32::MAX)?;
    let d_inv_sqrt = clamped_d.powf(-0.5)?; // [n, 1]

    let d_inv_mat = d_inv_sqrt.matmul(&d_inv_sqrt.t()?)?; // [n, n]
    let norm_a = a.mul(&d_inv_mat)?;

    let eye_n = Tensor::eye(n, candle_core::DType::F32, x.device())?;
    eye_n - norm_a
}
