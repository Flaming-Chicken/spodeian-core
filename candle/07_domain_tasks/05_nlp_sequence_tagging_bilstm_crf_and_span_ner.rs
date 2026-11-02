//! Nlp Sequence Tagging Bilstm Crf and Span Ner
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn logsumexp(x: &Tensor, dim: usize) -> Result<Tensor> {
    let max = x.max_keepdim(dim)?;
    let exp = (x.broadcast_sub(&max))?.exp()?;
    let sum_exp = exp.sum_keepdim(dim)?;
    sum_exp.log()?.broadcast_add(&max)?.squeeze(dim)
}

/// Linear-Chain CRF Partition Function (Log-Sum-Exp forward pass):
/// emissions: [B, T, K]
/// transitions: [K, K] where transitions[i, j] is score for transition i -> j.
/// Returns log Z: [B]
pub fn crf_log_partition(emissions: &Tensor, transitions: &Tensor) -> Result<Tensor> {
    let (b, t_len, k) = emissions.dims3()?;
    let mut alpha = emissions.narrow(1, 0, 1)?.squeeze(1)?; // [B, K]
    let trans = transitions.broadcast_as((b, k, k))?; // [B, K, K]

    for t in 1..t_len {
        let emit = emissions.narrow(1, t, 1)?; // [B, 1, K]
        let alpha_prev = alpha.unsqueeze(2)?; // [B, K, 1]
        let scores = alpha_prev.broadcast_add(&trans)?.broadcast_add(&emit)?; // [B, K, K]
        alpha = logsumexp(&scores, 1)?; // sum over previous state i -> [B, K]
    }

    logsumexp(&alpha, 1) // sum over last state k -> [B]
}

/// Score of gold tag sequence under CRF:
/// Score = sum_t emit(t, tag_t) + sum_t trans(tag_t, tag_{t+1})
pub fn crf_sequence_score(
    emissions: &Tensor,
    transitions: &Tensor,
    tags: &Tensor,
) -> Result<Tensor> {
    let (b, t_len, _k) = emissions.dims3()?;
    let emit_vec = emissions.to_vec3::<f32>()?;
    let trans_vec = transitions.to_vec2::<f32>()?;
    let tags_vec = tags.to_vec2::<f32>()?;

    let mut scores = vec![0.0f32; b];
    for bi in 0..b {
        let mut s = emit_vec[bi][0][tags_vec[bi][0] as usize];
        for t in 1..t_len {
            let prev = tags_vec[bi][t - 1] as usize;
            let curr = tags_vec[bi][t] as usize;
            s += emit_vec[bi][t][curr] + trans_vec[prev][curr];
        }
        scores[bi] = s;
    }

    Tensor::from_vec(scores, (b,), emissions.device())
}

/// CRF Negative Log-Likelihood Loss:
/// Loss = log Z - Score(y)
pub fn crf_nll_loss(emissions: &Tensor, transitions: &Tensor, tags: &Tensor) -> Result<Tensor> {
    let log_z = crf_log_partition(emissions, transitions)?;
    let gold_score = crf_sequence_score(emissions, transitions, tags)?;
    log_z.sub(&gold_score)
}

/// Span-based NER boundary feature representation:
/// span = [H[start]; H[end]; H[end] - H[start]]
pub fn span_endpoint_features(
    seq_embeds: &Tensor,
    start_idx: &Tensor,
    end_idx: &Tensor,
) -> Result<Tensor> {
    let (b, _t, d) = seq_embeds.dims3()?;
    let (_b2, s_len) = start_idx.dims2()?;
    let seq_vec = seq_embeds.to_vec3::<f32>()?;
    let start_vec = start_idx.to_vec2::<f32>()?;
    let end_vec = end_idx.to_vec2::<f32>()?;

    let mut out = vec![0.0f32; b * s_len * (3 * d)];
    for bi in 0..b {
        for si in 0..s_len {
            let s_idx = start_vec[bi][si] as usize;
            let e_idx = end_vec[bi][si] as usize;
            let row_offset = (bi * s_len + si) * (3 * d);
            for di in 0..d {
                let h_s = seq_vec[bi][s_idx][di];
                let h_e = seq_vec[bi][e_idx][di];
                out[row_offset + di] = h_s;
                out[row_offset + d + di] = h_e;
                out[row_offset + 2 * d + di] = h_e - h_s;
            }
        }
    }

    Tensor::from_vec(out, (b, s_len, 3 * d), seq_embeds.device())
}
