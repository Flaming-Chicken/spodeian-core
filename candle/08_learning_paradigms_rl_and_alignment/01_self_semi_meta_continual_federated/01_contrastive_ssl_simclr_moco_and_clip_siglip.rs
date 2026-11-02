//! Contrastive Ssl Simclr Moco and CLIP Siglip
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for L2 normalization along dimension 1.
fn normalize(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// Helper to compute cross-entropy where targets are the diagonal indices.
fn cross_entropy_diag(logits: &Tensor) -> Result<Tensor> {
    let log_sm = candle_nn::ops::log_softmax(logits, 1)?;
    let b = logits.dim(0)?;
    let vec = log_sm.to_vec2::<f32>()?;
    let mut sum_loss = 0.0f32;
    for i in 0..b {
        sum_loss += -vec[i][i];
    }
    Tensor::new(sum_loss / (b as f32), logits.device())
}

/// SimCLR NT-Xent Loss in Candle.
pub fn simclr_nt_xent_loss(z1: &Tensor, z2: &Tensor, temperature: f64) -> Result<Tensor> {
    let z1_norm = normalize(z1)?;
    let z2_norm = normalize(z2)?;
    let sim_matrix = ((z1_norm.matmul(&z2_norm.transpose(0, 1)?)?) * (1.0 / temperature))?;
    cross_entropy_diag(&sim_matrix)
}

/// MoCo Memory Queue Contrastive Loss in Candle.
pub fn moco_contrastive_loss(
    q: &Tensor,
    k_pos: &Tensor,
    queue: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    let q_norm = normalize(q)?;
    let k_norm = normalize(k_pos)?;
    let queue_norm = normalize(queue)?;

    let l_pos = ((q_norm.broadcast_mul(&k_norm)?.sum_keepdim(1)?) * (1.0 / temperature))?;
    let l_neg = ((q_norm.matmul(&queue_norm.transpose(0, 1)?)?) * (1.0 / temperature))?;
    let logits = Tensor::cat(&[&l_pos, &l_neg], 1)?;

    let log_sm = candle_nn::ops::log_softmax(&logits, 1)?;
    log_sm.narrow(1, 0, 1)?.mean_all()? * (-1.0)
}

/// CLIP Symmetric Cross-Entropy Loss in Candle.
pub fn clip_symmetric_loss(
    image_embeds: &Tensor,
    text_embeds: &Tensor,
    log_temp: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let i_norm = normalize(image_embeds)?;
    let t_norm = normalize(text_embeds)?;
    let temp_scalar = (log_temp.to_scalar::<f32>()?.min(4.6052f32)).exp() as f64;

    let logits = ((i_norm.matmul(&t_norm.transpose(0, 1)?)?) * temp_scalar)?.broadcast_add(bias)?;
    let loss_i2t = cross_entropy_diag(&logits)?;
    let loss_t2i = cross_entropy_diag(&logits.transpose(0, 1)?)?;
    (loss_i2t + loss_t2i)? * 0.5
}

/// SigLIP Pairwise Sigmoid Loss in Candle.
pub fn siglip_pairwise_loss(
    image_embeds: &Tensor,
    text_embeds: &Tensor,
    log_temp: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let b = image_embeds.dim(0)?;
    let i_norm = normalize(image_embeds)?;
    let t_norm = normalize(text_embeds)?;
    let temp_scalar = (log_temp.to_scalar::<f32>()?.min(4.6052f32)).exp() as f64;

    let logits = ((i_norm.matmul(&t_norm.transpose(0, 1)?)?) * temp_scalar)?.broadcast_add(bias)?;
    let l_vec = logits.to_vec2::<f32>()?;

    let mut total_loss = 0.0f32;
    for i in 0..b {
        for j in 0..b {
            let y = if i == j { 1.0f32 } else { -1.0f32 };
            let val = -y * l_vec[i][j];
            // softplus(val) = ln(1 + exp(val))
            let sp = (1.0f32 + val.exp()).ln();
            total_loss += sp;
        }
    }

    Tensor::new(total_loss / ((b * b) as f32), image_embeds.device())
}
