//! Active Learning Bald Coreset and Badge
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Bayesian Active Learning by Disagreement: H(mean(p)) - mean(H(p)).
pub fn bald_score(mc_probs: &Tensor, eps: f32) -> Result<Tensor> {
    let vec = mc_probs.to_vec3::<f32>()?;
    let m = vec.len();
    let b = vec[0].len();
    let c = vec[0][0].len();

    let mut out = Vec::with_capacity(b);

    for bi in 0..b {
        let mut mean_p = vec![0.0f32; c];
        for mi in 0..m {
            for ci in 0..c {
                mean_p[ci] += vec[mi][bi][ci];
            }
        }
        for ci in 0..c {
            mean_p[ci] /= m as f32;
        }

        let mut h_mean = 0.0f32;
        for ci in 0..c {
            h_mean += -mean_p[ci] * (mean_p[ci] + eps).ln();
        }

        let mut sum_h = 0.0f32;
        for mi in 0..m {
            let mut h_m = 0.0f32;
            for ci in 0..c {
                let p = vec[mi][bi][ci];
                h_m += -p * (p + eps).ln();
            }
            sum_h += h_m;
        }
        let mean_h = sum_h / (m as f32);

        out.push(h_mean - mean_h);
    }

    Tensor::from_vec(out, (b,), mc_probs.device())
}

/// Core-Set k-Center Greedy min distance from each unlabeled point to labeled points.
pub fn coreset_min_distances(unlabeled_feats: &Tensor, labeled_feats: &Tensor) -> Result<Tensor> {
    let u_vec = unlabeled_feats.to_vec2::<f32>()?;
    let l_vec = labeled_feats.to_vec2::<f32>()?;
    let nu = u_vec.len();
    let nl = l_vec.len();
    let d = u_vec[0].len();

    let mut out = Vec::with_capacity(nu);
    for i in 0..nu {
        let mut min_d = f32::MAX;
        for j in 0..nl {
            let mut sq = 0.0f32;
            for k in 0..d {
                let diff = u_vec[i][k] - l_vec[j][k];
                sq += diff * diff;
            }
            let dist = sq.sqrt();
            if dist < min_d {
                min_d = dist;
            }
        }
        out.push(min_d);
    }

    Tensor::from_vec(out, (nu,), unlabeled_feats.device())
}

/// BADGE gradient embeddings: g_i = (p_i - e_yhat) (x) x_i in R^(C x D), flattened to R^(C*D).
pub fn badge_gradient_embeddings(feats: &Tensor, probs: &Tensor) -> Result<Tensor> {
    let f_vec = feats.to_vec2::<f32>()?;
    let p_vec = probs.to_vec2::<f32>()?;
    let b = f_vec.len();
    let d = f_vec[0].len();
    let c = p_vec[0].len();

    let mut out = Vec::with_capacity(b * c * d);

    for bi in 0..b {
        let mut max_p = -1.0f32;
        let mut argmax = 0;
        for ci in 0..c {
            if p_vec[bi][ci] > max_p {
                max_p = p_vec[bi][ci];
                argmax = ci;
            }
        }

        for ci in 0..c {
            let y = if ci == argmax { 1.0f32 } else { 0.0f32 };
            let diff = p_vec[bi][ci] - y;
            for di in 0..d {
                out.push(diff * f_vec[bi][di]);
            }
        }
    }

    Tensor::from_vec(out, (b, c * d), feats.device())
}
