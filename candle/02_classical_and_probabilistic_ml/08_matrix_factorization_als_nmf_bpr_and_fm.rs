//! Matrix Factorization: ALS, NMF, BPR, and Factorization Machines in Candle.
//!
//! Features:
//! - Factorization Machine (FM) 2-way interactions evaluated in O(KD) time
//! - Latent low-rank matrix reconstruction P @ Q^T
//! - Bayesian Personalized Ranking (BPR) pairwise ranking loss

use candle_core::{Result, Tensor};

/// Evaluates 2-way Factorization Machine in O(KD) time:
/// y_hat = w0 + x @ w + 0.5 * sum_f ((x @ V)_f^2 - (x^2 @ V^2)_f)
pub fn factorization_machine_forward(
    x: &Tensor,
    w0: &Tensor,
    w: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let (n, _d) = x.dims2()?;

    // Linear terms: w0 + x @ w
    let w_col = if w.rank() == 1 {
        w.unsqueeze(1)?
    } else {
        w.clone()
    };
    let xw = x.matmul(&w_col)?.squeeze(1)?;
    let linear = (xw + w0.broadcast_as((n,))?)?;

    // 2-way interaction terms: 0.5 * sum_f ((x @ V)^2 - x^2 @ V^2)
    let xv = x.matmul(v)?; // [N, K]
    let xv_sq = xv.sqr()?; // [N, K]

    let x_sq = x.sqr()?; // [N, D]
    let v_sq = v.sqr()?; // [D, K]
    let x_sq_v_sq = x_sq.matmul(&v_sq)?; // [N, K]

    let diff = (&xv_sq - &x_sq_v_sq)?;
    let interaction = diff.sum_keepdim(1)?.squeeze(1)?.affine(0.5, 0.0)?;

    &linear + &interaction
}

/// Computes low-rank rating prediction: R_hat = P @ Q^T.
pub fn matrix_factorization_predict(p: &Tensor, q: &Tensor) -> Result<Tensor> {
    p.matmul(&q.t()?)
}

/// Evaluates Bayesian Personalized Ranking (BPR) loss: -mean(log(sigmoid(p_u @ q_pos - p_u @ q_neg)))
pub fn bpr_loss(p_u: &Tensor, q_pos: &Tensor, q_neg: &Tensor) -> Result<Tensor> {
    let x_pos = (p_u * q_pos)?.sum_keepdim(1)?.squeeze(1)?;
    let x_neg = (p_u * q_neg)?.sum_keepdim(1)?.squeeze(1)?;
    let diff = (x_pos - x_neg)?;
    let sig = candle_nn::ops::sigmoid(&diff)?;
    let eps = 1e-8f64;
    let log_sig = (sig.affine(1.0, eps))?.log()?;
    log_sig.affine(-1.0, 0.0)?.mean_all()
}
