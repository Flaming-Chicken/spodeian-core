//! Meta Learning Prototypical MAML and Reptile
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Compute class prototypes as mean support embeddings across shots: [N, K, D] -> [N, D].
pub fn protonet_prototypes(support_embeds: &Tensor) -> Result<Tensor> {
    support_embeds.mean(1)
}

/// Compute negative squared Euclidean distances between queries and prototypes: [Q, N].
pub fn protonet_logits(query_embeds: &Tensor, prototypes: &Tensor) -> Result<Tensor> {
    let q_vec = query_embeds.to_vec2::<f32>()?;
    let c_vec = prototypes.to_vec2::<f32>()?;
    let q = q_vec.len();
    let n = c_vec.len();
    let d = q_vec[0].len();

    let mut out = Vec::with_capacity(q * n);
    for i in 0..q {
        for j in 0..n {
            let mut sum_sq = 0.0f32;
            for k in 0..d {
                let diff = q_vec[i][k] - c_vec[j][k];
                sum_sq += diff * diff;
            }
            out.push(-sum_sq);
        }
    }

    Tensor::from_vec(out, (q, n), query_embeds.device())
}

/// Single inner gradient adaptation step for linear regression under MSE loss.
pub fn maml_inner_linear_step(
    x_s: &Tensor,
    y_s: &Tensor,
    w: &Tensor,
    b: &Tensor,
    alpha: f64,
) -> Result<(Tensor, Tensor)> {
    let pred_s = x_s.matmul(w)?.broadcast_add(b)?;
    let err = (pred_s - y_s)?;
    let n_elements = err.elem_count() as f64;
    let scale = 2.0 / n_elements;

    let grad_w = (x_s.transpose(0, 1)?.matmul(&err)? * scale)?;
    let grad_b = (err.sum_keepdim(0)? * scale)?.reshape(&[b.dim(0)?])?;

    let w_adapted = (w - (grad_w * alpha)?)?;
    let b_adapted = (b - (grad_b * alpha)?)?;
    Ok((w_adapted, b_adapted))
}

/// Reptile outer meta-update: theta + epsilon * (theta_adapted - theta).
pub fn reptile_outer_step(theta: &Tensor, theta_adapted: &Tensor, epsilon: f64) -> Result<Tensor> {
    let diff = (theta_adapted - theta)?;
    theta + (diff * epsilon)?
}
