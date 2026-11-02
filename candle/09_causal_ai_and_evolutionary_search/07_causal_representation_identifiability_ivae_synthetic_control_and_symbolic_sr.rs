//! Causal Representation Identifiability iVAE Synthetic Control and Symbolic Sr
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Closed-form KL divergence between posterior q(z|x, u) and conditional prior p(z|u).
pub fn ivae_kl_divergence(
    mu_z: &Tensor,
    logvar_z: &Tensor,
    mu_u: &Tensor,
    logvar_u: &Tensor,
) -> Result<Tensor> {
    let var_z = logvar_z.exp()?;
    let var_u = logvar_u.exp()?;

    let diff_mu = mu_z.sub(mu_u)?;
    let sq_diff = diff_mu.mul(&diff_mu)?;
    let num = var_z.add(&sq_diff)?;
    let ratio = num.div(&var_u)?;

    let logvar_diff = logvar_z.sub(logvar_u)?;
    let term = ratio.affine(1.0, -1.0)?.sub(&logvar_diff)?;
    let kl_per_dim = term.affine(0.5, 0.0)?;

    kl_per_dim.sum(1)
}

/// Euclidean projection onto the probability simplex {w >= 0, sum(w) = 1}.
pub fn project_simplex(v: &Tensor) -> Result<Tensor> {
    let mut u = v.flatten_all()?.to_vec1::<f32>()?;
    let n = u.len();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut theta = 0.0f32;

    for j in 1..=n {
        cumsum += u[j - 1];
        let test_val = u[j - 1] - (cumsum - 1.0) / (j as f32);
        if test_val > 0.0 {
            theta = (cumsum - 1.0) / (j as f32);
        }
    }

    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
    let mut w = Vec::with_capacity(n);
    for val in v_vec {
        w.push((val - theta).max(0.0));
    }

    Tensor::from_vec(w, (n,), v.device())
}

/// Compute synthetic counterfactual control trajectory and treatment effect.
pub fn synthetic_control_effect(y1: &Tensor, y0: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let y1_synth = y0.matmul(&w.unsqueeze(1)?)?.squeeze(1)?;
    let tau = y1.sub(&y1_synth)?;
    Ok((y1_synth, tau))
}

/// Symbolic regression Pareto multi-objective loss: MSE(y, y_hat) + lambda * complexity.
pub fn symbolic_pareto_loss(
    y_true: &Tensor,
    y_pred: &Tensor,
    complexity: &Tensor,
    lambda_reg: &Tensor,
) -> Result<Tensor> {
    let diff = y_true.sub(y_pred)?;
    let mse = diff.mul(&diff)?.mean_all()?;
    let reg = complexity.mul(lambda_reg)?;
    mse.add(&reg)
}
