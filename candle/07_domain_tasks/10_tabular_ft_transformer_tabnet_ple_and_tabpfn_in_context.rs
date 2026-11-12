//! Tabular Ft Transformer Tabnet Ple and Tabpfn In Context
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// FT-Transformer Feature Tokenizer:
/// Projects numerical features and concatenates with categorical and CLS tokens.
pub fn ft_transformer_tokenize(
    x_num: &Tensor,
    w_num: &Tensor,
    b_num: &Tensor,
    cat_tokens: &Tensor,
    cls_token: &Tensor,
) -> Result<Tensor> {
    let (b, _n_num) = x_num.dims2()?;
    let (_n_cat, d) = (cat_tokens.dim(1)?, cat_tokens.dim(2)?);

    let num_tokens = x_num
        .unsqueeze(2)?
        .broadcast_mul(&w_num.unsqueeze(0)?)?
        .broadcast_add(&b_num.unsqueeze(0)?)?;

    let cls_expanded = cls_token.broadcast_as((b, 1, d))?;
    Tensor::cat(&[&cls_expanded, &num_tokens, cat_tokens], 1)
}

/// Exact Sparsemax activation projection onto probability simplex.
pub fn sparsemax(z: &Tensor) -> Result<Tensor> {
    let (b, d) = z.dims2()?;
    let z_vec = z.to_vec2::<f32>()?;
    let mut out_vec = Vec::with_capacity(b);

    for row in z_vec {
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut k_z = 1usize;
        let mut cumsum_k = 0.0f32;

        for (idx, &val) in sorted.iter().enumerate() {
            let k = idx + 1;
            cumsum += val;
            if 1.0 + (k as f32) * val > cumsum {
                k_z = k;
                cumsum_k = cumsum;
            }
        }

        let tau = (cumsum_k - 1.0) / (k_z as f32);
        let out_row: Vec<f32> = row.iter().map(|&v| (v - tau).max(0.0)).collect();
        out_vec.push(out_row);
    }

    Tensor::new(out_vec, z.device())?.reshape((b, d))
}

/// TabNet Attentive Transformer feature selection mask and prior update.
pub fn tabnet_attentive_mask(
    p_prior: &Tensor,
    h_features: &Tensor,
    gamma: f64,
) -> Result<(Tensor, Tensor)> {
    let logits = (p_prior * h_features)?;
    let mask = sparsemax(&logits)?;
    let p_next = (p_prior * (gamma - &mask)?)?;
    Ok((mask, p_next))
}

/// Progressive Layered Extraction (PLE) gated task aggregation.
pub fn ple_task_routing(
    expert_outputs: &Tensor,
    gate_logits: &Tensor,
    expert_indices: &[u32],
) -> Result<Tensor> {
    let weights = candle_nn::ops::softmax(gate_logits, 1)?.unsqueeze(2)?;
    let mut cand_vec = Vec::with_capacity(expert_indices.len());
    for &idx in expert_indices {
        cand_vec.push(expert_outputs.narrow(1, idx as usize, 1)?);
    }
    let cand_refs: Vec<&Tensor> = cand_vec.iter().collect();
    let candidates = Tensor::cat(&cand_refs, 1)?;
    candidates.broadcast_mul(&weights)?.sum(1)
}
