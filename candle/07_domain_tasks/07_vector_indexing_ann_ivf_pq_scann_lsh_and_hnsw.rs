//! Vector Indexing Ann IVF PQ Scann Lsh and HNSW
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Product Quantization (PQ) Asymmetric Distance Table:
/// query_subvectors: [B, M, d]
/// codebooks: [M, K, d]
/// Returns squared L2 distance table: [B, M, K]
pub fn pq_distance_table(query_subvectors: &Tensor, codebooks: &Tensor) -> Result<Tensor> {
    let q_4d = query_subvectors.unsqueeze(2)?;
    let c_4d = codebooks.unsqueeze(0)?;
    let diff = q_4d.broadcast_sub(&c_4d)?;
    diff.sqr()?.sum(3)
}

/// Asymmetric Distance Computation (ADC) via table lookup:
/// dist_table: [B, M, K]
/// codes: [B, N_docs, M] with integer cluster indices in [0, K-1]
pub fn pq_asymmetric_distance(dist_table: &Tensor, codes: &Tensor) -> Result<Tensor> {
    let (b, n_docs, m) = codes.dims3()?;
    let dt_vec = dist_table.to_vec3::<f32>()?;
    let codes_vec = codes.to_vec3::<f32>()?;

    let mut out = vec![0.0f32; b * n_docs];
    for bi in 0..b {
        for ni in 0..n_docs {
            let mut sum_d = 0.0f32;
            for mi in 0..m {
                let c = codes_vec[bi][ni][mi] as usize;
                sum_d += dt_vec[bi][mi][c];
            }
            out[bi * n_docs + ni] = sum_d;
        }
    }

    Tensor::from_vec(out, (b, n_docs), dist_table.device())
}

/// ScaNN Anisotropic Vector Quantization Distortion Loss (Guo et al.):
/// L = ||e_perp||^2 + (1 + omega) * ||e_parallel||^2
pub fn scann_anisotropic_loss(
    x: &Tensor,
    x_tilde: &Tensor,
    omega: f64,
    eps: f64,
) -> Result<Tensor> {
    let e = (x - x_tilde)?;
    let x_norm_sq = (x.sqr()?.sum_keepdim(1)? + eps)?;
    let dot_ex = e.mul(x)?.sum_keepdim(1)?;
    let scale = dot_ex.div(&x_norm_sq)?;
    let e_parallel = scale.broadcast_mul(x)?;
    let e_perp = e.sub(&e_parallel)?;

    let loss_perp = e_perp.sqr()?.sum(1)?;
    let loss_parallel = e_parallel.sqr()?.sum(1)?;
    let scaled_parallel = (loss_parallel * (1.0 + omega))?;
    loss_perp.add(&scaled_parallel)
}

/// Locality-Sensitive Hashing (LSH) Random Projection Cosine Similarity:
/// b_q = sign(q @ R), b_d = sign(d @ R)
pub fn lsh_hash_hamming_distance(q: &Tensor, d: &Tensor, hyperplanes: &Tensor) -> Result<Tensor> {
    let q_proj = q.matmul(hyperplanes)?.to_vec2::<f32>()?;
    let d_proj = d.matmul(hyperplanes)?.to_vec2::<f32>()?;
    let b = q_proj.len();
    let n_bits = q_proj[0].len();

    let mut out = vec![0.0f32; b];
    for bi in 0..b {
        let mut sum_val = 0.0f32;
        for ni in 0..n_bits {
            let bq = if q_proj[bi][ni] >= 0.0 { 1.0 } else { -1.0 };
            let bd = if d_proj[bi][ni] >= 0.0 { 1.0 } else { -1.0 };
            sum_val += bq * bd;
        }
        out[bi] = sum_val / (n_bits as f32);
    }

    Tensor::from_vec(out, (b,), q.device())
}
