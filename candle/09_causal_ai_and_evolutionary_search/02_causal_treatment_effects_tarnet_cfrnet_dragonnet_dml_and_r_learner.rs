//! Causal Treatment Effects TARNet CFRNet Dragonnet DML and R Learner
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Treatment-Agnostic Representation Network (TARNet):
/// phi = ReLU(x @ phi_w + phi_b)
/// y0 = phi @ h0_w + h0_b
/// y1 = phi @ h1_w + h1_b
/// tau = y1 - y0
pub fn tarnet_forward(
    x: &Tensor,
    phi_w: &Tensor,
    phi_b: &Tensor,
    h0_w: &Tensor,
    h0_b: &Tensor,
    h1_w: &Tensor,
    h1_b: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let phi = x.matmul(phi_w)?.broadcast_add(phi_b)?.relu()?;
    let y0 = phi.matmul(h0_w)?.broadcast_add(h0_b)?;
    let y1 = phi.matmul(h1_w)?.broadcast_add(h1_b)?;
    let tau = (&y1 - &y0)?;
    Ok((phi, y0, y1, tau))
}

/// CFRNet linear MMD representation balance penalty:
/// MMD^2 = || mu_1 - mu_0 ||_2^2
pub fn cfrnet_mmd_loss(phi: &Tensor, t: &Tensor) -> Result<Tensor> {
    let (n, h) = phi.dims2()?;
    let phi_vec = phi.to_vec2::<f32>()?;
    let t_vec = t.to_vec2::<f32>()?;

    let mut n_treated = 0.0f32;
    let mut n_control = 0.0f32;
    let mut sum_1 = vec![0.0f32; h];
    let mut sum_0 = vec![0.0f32; h];

    for i in 0..n {
        let is_t = t_vec[i][0] > 0.5;
        if is_t {
            n_treated += 1.0;
            for j in 0..h {
                sum_1[j] += phi_vec[i][j];
            }
        } else {
            n_control += 1.0;
            for j in 0..h {
                sum_0[j] += phi_vec[i][j];
            }
        }
    }

    n_treated = n_treated.max(1.0);
    n_control = n_control.max(1.0);

    let mut mmd_sq = 0.0f32;
    for j in 0..h {
        let diff = (sum_1[j] / n_treated) - (sum_0[j] / n_control);
        mmd_sq += diff * diff;
    }

    Tensor::from_vec(vec![mmd_sq], 1, phi.device())
}

/// Dragonnet propensity head and BCE loss:
/// e = sigmoid(phi @ w_prop + b_prop)
/// loss = -1/N sum [t * log(e) + (1-t) * log(1-e)]
pub fn dragonnet_propensity_loss(
    phi: &Tensor,
    t: &Tensor,
    w_prop: &Tensor,
    b_prop: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (n, _) = phi.dims2()?;
    let logits = phi.matmul(w_prop)?.broadcast_add(b_prop)?;
    let logit_vec = logits.to_vec2::<f32>()?;
    let t_vec = t.to_vec2::<f32>()?;

    let mut prop_out = Vec::with_capacity(n);
    let mut bce_sum = 0.0f32;

    for i in 0..n {
        let z = logit_vec[i][0];
        let p = 1.0f32 / (1.0f32 + (-z).exp());
        let p_c = p.clamp(eps, 1.0 - eps);
        let ti = t_vec[i][0];
        bce_sum += -(ti * p_c.ln() + (1.0 - ti) * (1.0 - p_c).ln());
        prop_out.push(p);
    }

    let prop_t = Tensor::from_vec(prop_out, (n, 1), phi.device())?;
    let bce_t = Tensor::from_vec(vec![bce_sum / (n as f32)], 1, phi.device())?;
    Ok((prop_t, bce_t))
}

/// Double Machine Learning (DML) Robinson CATE estimator:
/// tau = sum_i (T_i - e_hat_i)(Y_i - m_hat_i) / sum_i (T_i - e_hat_i)^2
pub fn dml_cate_estimate(y: &Tensor, t: &Tensor, m_hat: &Tensor, e_hat: &Tensor) -> Result<Tensor> {
    let (n, _) = y.dims2()?;
    let y_vec = y.to_vec2::<f32>()?;
    let t_vec = t.to_vec2::<f32>()?;
    let m_vec = m_hat.to_vec2::<f32>()?;
    let e_vec = e_hat.to_vec2::<f32>()?;

    let mut num = 0.0f32;
    let mut den = 0.0f32;

    for i in 0..n {
        let res_t = t_vec[i][0] - e_vec[i][0];
        let res_y = y_vec[i][0] - m_vec[i][0];
        num += res_t * res_y;
        den += res_t * res_t;
    }

    let tau = num / den.max(1e-8);
    Tensor::from_vec(vec![tau], 1, y.device())
}
