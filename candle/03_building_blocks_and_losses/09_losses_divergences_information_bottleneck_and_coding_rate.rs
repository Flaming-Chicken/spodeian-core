//! Losses: Divergences, Information Bottleneck, and Coding Rate (C03 Topic 09).
//!
//! Native Rust Candle implementation of:
//! 1. Forward & Reverse KL Divergence + JSD.
//! 2. alpha-Renyi Divergence.
//! 3. Itakura-Saito Distance.
//! 4. Variational Information Bottleneck (VIB) Gaussian Prior KL.
//! 5. Maximal Coding Rate Reduction (MCR^2) Total Rate.
//! 6. Linear Anchor.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// Forward KL Divergence: D_KL(P || Q) = sum P * log(P / Q).
pub fn kl_forward(p: &Tensor, q: &Tensor, eps: f64) -> Result<Tensor> {
    let p_vec = p.to_vec2::<f32>()?;
    let q_vec = q.to_vec2::<f32>()?;
    let b = p_vec.len();
    let k = p_vec[0].len();
    let eps_f = eps as f32;

    let mut total = 0.0f32;
    for i in 0..b {
        for j in 0..k {
            let p_val = p_vec[i][j].max(eps_f);
            let q_val = q_vec[i][j].max(eps_f);
            total += p_val * (p_val.ln() - q_val.ln());
        }
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], p.device())
}

/// Reverse KL Divergence: D_KL(Q || P) = sum Q * log(Q / P).
pub fn kl_reverse(p: &Tensor, q: &Tensor, eps: f64) -> Result<Tensor> {
    kl_forward(q, p, eps)
}

/// Jensen-Shannon Divergence: 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), M = 0.5 * (P + Q).
pub fn jsd_forward(p: &Tensor, q: &Tensor) -> Result<Tensor> {
    let m = ((p + q)? * 0.5)?;
    let kl_pm = kl_forward(p, &m, 1e-12)?;
    let kl_qm = kl_forward(q, &m, 1e-12)?;
    (kl_pm + kl_qm)? * 0.5
}

/// alpha-Renyi Divergence: 1/(alpha - 1) * log sum P^alpha * Q^(1-alpha).
pub fn renyi_divergence_forward(p: &Tensor, q: &Tensor, alpha: f64, eps: f64) -> Result<Tensor> {
    let p_vec = p.to_vec2::<f32>()?;
    let q_vec = q.to_vec2::<f32>()?;
    let b = p_vec.len();
    let k = p_vec[0].len();

    let alpha_f = alpha as f32;
    let eps_f = eps as f32;
    let scale = 1.0f32 / (alpha_f - 1.0f32);

    let mut total = 0.0f32;
    for i in 0..b {
        let mut inner = 0.0f32;
        for j in 0..k {
            let p_val = p_vec[i][j].max(eps_f);
            let q_val = q_vec[i][j].max(eps_f);
            inner += p_val.powf(alpha_f) * q_val.powf(1.0f32 - alpha_f);
        }
        total += scale * inner.max(eps_f).ln();
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], p.device())
}

/// Itakura-Saito Distance: mean(x/y - log(x/y) - 1).
pub fn itakura_saito_forward(x: &Tensor, y: &Tensor, eps: f64) -> Result<Tensor> {
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let y_vec = y.flatten_all()?.to_vec1::<f32>()?;
    let n = x_vec.len();
    let eps_f = eps as f32;

    let mut total = 0.0f32;
    for i in 0..n {
        let xv = x_vec[i].max(eps_f);
        let yv = y_vec[i].max(eps_f);
        let ratio = xv / yv;
        total += ratio - ratio.ln() - 1.0f32;
    }
    let mean = total / (n as f32);
    Tensor::new(&[mean], x.device())
}

/// Variational Information Bottleneck (VIB) Gaussian Prior KL.
pub fn vib_kl_forward(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let mu_vec = mu.to_vec2::<f32>()?;
    let lv_vec = logvar.to_vec2::<f32>()?;
    let b = mu_vec.len();
    let d = mu_vec[0].len();

    let mut total = 0.0f32;
    for i in 0..b {
        let mut sum_kl = 0.0f32;
        for j in 0..d {
            let m = mu_vec[i][j];
            let s = lv_vec[i][j];
            sum_kl += -0.5f32 * (1.0f32 + s - m * m - s.exp());
        }
        total += sum_kl;
    }
    let mean = total / (b as f32);
    Tensor::new(&[mean], mu.device())
}

/// Cholesky log-determinant for symmetric positive definite matrix A.
fn cholesky_logdet(a: &[f32], n: usize) -> f32 {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                l[i * n + j] = diag.max(1e-12).sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j].max(1e-12);
            }
        }
    }
    let mut logdet = 0.0f32;
    for i in 0..n {
        logdet += 2.0f32 * l[i * n + i].ln();
    }
    logdet
}

/// Maximal Coding Rate Reduction (MCR^2) Total Rate: 0.5 * log det(I + D / (N * eps^2) * Z^T Z).
pub fn mcr2_coding_rate_forward(z: &Tensor, eps: f64) -> Result<Tensor> {
    let z_vec = z.to_vec2::<f32>()?;
    let n = z_vec.len();
    let d = z_vec[0].len();

    let scale = (d as f32) / ((n as f32) * (eps as f32).powi(2));
    let mut c = vec![0.0f32; d * d];

    for i in 0..d {
        c[i * d + i] = 1.0f32;
        for j in 0..d {
            let mut dot = 0.0f32;
            for row in 0..n {
                dot += z_vec[row][i] * z_vec[row][j];
            }
            c[i * d + j] += scale * dot;
        }
    }

    let logdet = cholesky_logdet(&c, d);
    let rate = 0.5f32 * logdet;
    Tensor::new(&[rate], z.device())
}
