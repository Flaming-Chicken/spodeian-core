//! Recommenders Dlrm Dcnv2 Sasrec and Tiger Semantic Ids
//! in candle/07_domain_tasks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// DLRM Explicit Pairwise Dot-Product Feature Interaction.
pub fn dlrm_feature_interaction(v_dense: &Tensor, v_sparse: &Tensor) -> Result<Tensor> {
    let s = v_sparse.dim(1)?;
    let v_all = Tensor::cat(&[&v_dense.unsqueeze(1)?, v_sparse], 1)?;
    let mat = v_all.matmul(&v_all.transpose(1, 2)?)?;

    let total = s + 1;
    let mut pairs = Vec::with_capacity(total * (total - 1) / 2);
    for i in 1..total {
        for j in 0..i {
            let elem = mat.narrow(1, i, 1)?.narrow(2, j, 1)?.squeeze(2)?;
            pairs.push(elem);
        }
    }
    let pair_refs: Vec<&Tensor> = pairs.iter().collect();
    let interactions = Tensor::cat(&pair_refs, 1)?;
    Tensor::cat(&[v_dense, &interactions], 1)
}

/// DCN-v2 (Deep & Cross Network v2) Cross Layer.
pub fn dcn_v2_cross_layer(
    x0: &Tensor,
    xl: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let proj = xl.matmul(weight)?.broadcast_add(bias)?;
    x0.broadcast_mul(&proj)? + xl
}

/// SASRec Causally-Masked Self-Attention for Sequential Recommendation.
pub fn sasrec_causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_b, t, d) = q.dims3()?;
    let scale = 1.0 / (d as f64).sqrt();
    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;

    // Create causal upper-triangular mask
    let mut mask_vec = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask_vec[i * t + j] = -1e9f32;
            }
        }
    }
    let mask_t = Tensor::new(mask_vec, q.device())?.reshape((1, t, t))?;
    let masked_scores = scores.broadcast_add(&mask_t)?;
    let weights = candle_nn::ops::softmax(&masked_scores, 2)?;
    weights.matmul(v)
}

/// TIGER / RQ-VAE Residual Vector Quantization Step.
pub fn rq_vae_quantize_step(
    residual: &Tensor,
    codebook: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, _d) = residual.dims2()?;
    let r_sq = residual.sqr()?.sum_keepdim(1)?;
    let c_sq = codebook.sqr()?.sum_keepdim(1)?.transpose(0, 1)?;
    let r_c = residual.matmul(&codebook.transpose(0, 1)?)?;
    let dist = (r_sq.broadcast_sub(&(r_c * 2.0)?)?.broadcast_add(&c_sq))?;

    let dist_vec = dist.to_vec2::<f32>()?;
    let mut indices_vec = Vec::with_capacity(b);
    let mut quant_vec = Vec::with_capacity(b);

    for row in dist_vec {
        let mut min_idx = 0usize;
        let mut min_val = f32::INFINITY;
        for (idx, &v) in row.iter().enumerate() {
            if v < min_val {
                min_val = v;
                min_idx = idx;
            }
        }
        indices_vec.push(min_idx as i64);
        quant_vec.push(codebook.narrow(0, min_idx, 1)?);
    }

    let quant_refs: Vec<&Tensor> = quant_vec.iter().collect();
    let quantized = Tensor::cat(&quant_refs, 0)?;
    let indices = Tensor::new(indices_vec, residual.device())?;
    let next_residual = (residual - &quantized)?;

    Ok((quantized, indices, next_residual))
}
