//! Nlp Dense Retrieval Colbert Ubinary and RAG
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// ColBERT Multi-Vector Late Interaction (MaxSim):
/// Score(Q, D) = sum_i max_j (q_i . d_j)
pub fn colbert_maxsim(query_embeds: &Tensor, doc_embeds: &Tensor) -> Result<Tensor> {
    let d_t = doc_embeds.transpose(1, 2)?;
    let sims = query_embeds.matmul(&d_t)?;
    let max_per_q = sims.max(2)?;
    max_per_q.sum(1)
}

/// 1-Bit Unsigned Binary (UBinary) Quantization & Similarity:
/// b = sign(x) in {-1, +1}
/// Dot product: sum_k (b_q * b_d)
pub fn ubinary_quantize_and_dot(q_float: &Tensor, d_float: &Tensor) -> Result<Tensor> {
    let q_vec = q_float.to_vec2::<f32>()?;
    let d_vec = d_float.to_vec2::<f32>()?;
    let b = q_vec.len();
    let dim = q_vec[0].len();

    let mut dots = vec![0.0f32; b];
    for bi in 0..b {
        let mut sum_val = 0.0f32;
        for k in 0..dim {
            let qb = if q_vec[bi][k] >= 0.0 { 1.0 } else { -1.0 };
            let db = if d_vec[bi][k] >= 0.0 { 1.0 } else { -1.0 };
            sum_val += qb * db;
        }
        dots[bi] = sum_val;
    }

    Tensor::from_vec(dots, (b,), q_float.device())
}

/// Reciprocal Rank Fusion (RRF) for hybrid sparse-dense retrieval:
/// RRF(d) = 1 / (k + rank1) + 1 / (k + rank2)
pub fn reciprocal_rank_fusion(ranks1: &Tensor, ranks2: &Tensor, k: f64) -> Result<Tensor> {
    let term1 = (ranks1 + k)?.recip()?;
    let term2 = (ranks2 + k)?.recip()?;
    term1.add(&term2)
}

/// In-Batch Contrastive InfoNCE Loss for Dense Bi-Encoders:
/// L = -log( exp(q_i . p_i / tau) / sum_j exp(q_i . p_j / tau) )
pub fn dense_infonce_loss(query_vecs: &Tensor, pos_vecs: &Tensor, tau: f64) -> Result<Tensor> {
    let (b, _d) = query_vecs.dims2()?;
    let p_t = pos_vecs.t()?;
    let logits = (query_vecs.matmul(&p_t)? / tau)?;
    let log_sm = candle_nn::ops::log_softmax(&logits, 1)?;
    let log_sm_vec = log_sm.to_vec2::<f32>()?;
    let mut diag_sum = 0.0f32;
    for i in 0..b {
        diag_sum -= log_sm_vec[i][i];
    }
    Tensor::new(diag_sum / (b as f32), query_vecs.device())
}
