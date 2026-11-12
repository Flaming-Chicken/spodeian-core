//! AI Analysis Representation Similarity CKA SVCCA RSA and Intrinsic Dim
//! in candle/10_xai_and_ai_analysis_methods.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn center_gram(g: &Tensor) -> Result<Tensor> {
    let mean_row = g.mean_keepdim(0)?;
    let mean_col = g.mean_keepdim(1)?;
    let mean_all = g.mean_all()?;
    let g1 = g.broadcast_sub(&mean_row)?;
    let g2 = g1.broadcast_sub(&mean_col)?;
    g2.broadcast_add(&mean_all)
}

/// Linear Centered Kernel Alignment (CKA — Kornblith et al., 2019).
pub fn linear_cka(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let k = x.matmul(&x.t()?)?;
    let l_mat = y.matmul(&y.t()?)?;

    let k_c = center_gram(&k)?;
    let l_c = center_gram(&l_mat)?;

    let hsic_xy = k_c.mul(&l_c)?.sum_all()?.to_scalar::<f32>()?;
    let hsic_xx = k_c.mul(&k_c)?.sum_all()?.to_scalar::<f32>()?;
    let hsic_yy = l_c.mul(&l_c)?.sum_all()?.to_scalar::<f32>()?;

    let denom = (hsic_xx * hsic_yy).sqrt() + 1e-8;
    Tensor::new(hsic_xy / denom, x.device())
}

/// Representational Similarity Analysis Pearson correlation on upper-triangular RDM entries.
pub fn rsa_correlation(rdm_a: &Tensor, rdm_b: &Tensor) -> Result<Tensor> {
    let n = rdm_a.dim(0)?;
    let a_vec = rdm_a.to_vec2::<f32>()?;
    let b_vec = rdm_b.to_vec2::<f32>()?;

    let mut va = Vec::new();
    let mut vb = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            va.push(a_vec[i][j]);
            vb.push(b_vec[i][j]);
        }
    }

    let mean_a = va.iter().sum::<f32>() / (va.len() as f32);
    let mean_b = vb.iter().sum::<f32>() / (vb.len() as f32);

    let mut num = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for k in 0..va.len() {
        let da = va[k] - mean_a;
        let db = vb[k] - mean_b;
        num += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let rsa = num / ((var_a * var_b).sqrt() + 1e-8);
    Tensor::new(rsa, rdm_a.device())
}

/// TwoNN intrinsic manifold dimension estimator (Facco et al., 2017).
pub fn twonn_intrinsic_dimension(r1: &Tensor, r2: &Tensor) -> Result<Tensor> {
    let v1 = r1.to_vec1::<f32>()?;
    let v2 = r2.to_vec1::<f32>()?;
    let n = v1.len() as f32;

    let mut sum_log = 0.0f32;
    for i in 0..v1.len() {
        let mu = v2[i] / (v1[i] + 1e-8);
        sum_log += mu.ln();
    }

    let id = n / (sum_log + 1e-8);
    Tensor::new(id, r1.device())
}
