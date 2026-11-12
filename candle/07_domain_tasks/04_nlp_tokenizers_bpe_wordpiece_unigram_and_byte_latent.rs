//! Nlp Tokenizers Bpe Wordpiece Unigram and Byte Latent
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// WordPiece Mutual Information Pair Score:
/// S_{ij} = C_{ij} / (C_i * C_j + eps)
pub fn wordpiece_pair_score(
    bigram_counts: &Tensor,
    unigram_counts: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let u_col = unigram_counts.unsqueeze(1)?;
    let u_row = unigram_counts.unsqueeze(0)?;
    let outer = u_col.broadcast_mul(&u_row)?;
    let denom = (outer + eps)?;
    bigram_counts.div(&denom)
}

/// Byte-Level Model Patch Aggregator (MegaByte / BLT):
/// Average pools byte embeddings across patch_size dimension.
pub fn byte_patch_pooling(byte_embeddings: &Tensor, patch_size: usize) -> Result<Tensor> {
    let (b, t, d) = byte_embeddings.dims3()?;
    let num_patches = t / patch_size;
    let reshaped = byte_embeddings.reshape((b, num_patches, patch_size, d))?;
    let sum_p = reshaped.sum(2)?;
    sum_p / (patch_size as f64)
}

/// Local Byte Entropy for Dynamic Boundary Detection (BLT):
/// H = -sum_v p_v * log2(p_v + eps)
pub fn entropy_boundary_detector(probs: &Tensor, eps: f64) -> Result<Tensor> {
    let ln2 = std::f64::consts::LN_2;
    let probs_eps = (probs + eps)?;
    let log2_p = (probs_eps.log()? / ln2)?;
    let p_log_p = probs.mul(&log2_p)?;
    let sum_ent = p_log_p.sum(2)?;
    sum_ent.neg()
}
