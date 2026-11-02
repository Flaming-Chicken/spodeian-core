//! Candle Implementation for Category 04 Topic 19 (Subfolder 03 Topic 05):
//! Non-Backprop: Forward-Forward, Predictive Coding, EqProp, DFA, and Hebbian Learning.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Forward-Forward algorithm: goodness loss and layer normalization (Hinton, 2022).
//! 2. Predictive Coding: representation error minimization step (Rao & Ballard, 1999).
//! 3. Direct Feedback Alignment (DFA): fixed random feedback projection (Nøkland, 2016).
//! 4. Oja's Hebbian Rule: normalized local PCA update (Oja, 1982).
//! 5. Linear anchor evaluation for transitive multi-track parity.

use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Linear Anchor module for cross-track parity verification.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xw = x.matmul(&self.weight)?;
        xw.broadcast_add(&self.bias)
    }
}

/// Forward-Forward algorithm goodness loss and layer normalization (Hinton, 2022).
///
/// loss = 1/B sum_i [softplus(theta - ||h_pos||^2) + softplus(||h_neg||^2 - theta)]
/// h_norm = h_pos / (||h_pos||_2 + eps)
pub fn forward_forward_loss_and_norm(
    h_pos: &Tensor,
    h_neg: &Tensor,
    theta: f64,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let (b_size, d_size) = h_pos.dims2()?;
    let dev = h_pos.device();

    let hp_vec = h_pos.to_vec2::<f32>()?;
    let hn_vec = h_neg.to_vec2::<f32>()?;

    let mut total_loss = 0.0f32;
    let mut norm_pos_matrix = vec![vec![0.0f32; d_size]; b_size];

    let th = theta as f32;
    let eps_f = eps as f32;

    for i in 0..b_size {
        let mut g_p = 0.0f32;
        let mut g_n = 0.0f32;
        for j in 0..d_size {
            g_p += hp_vec[i][j] * hp_vec[i][j];
            g_n += hn_vec[i][j] * hn_vec[i][j];
        }

        // softplus(x) = ln(1 + exp(x))
        let loss_p = (th - g_p).exp().ln_1p();
        let loss_n = (g_n - th).exp().ln_1p();
        total_loss += loss_p + loss_n;

        let norm_p = (g_p).sqrt() + eps_f;
        for j in 0..d_size {
            norm_pos_matrix[i][j] = hp_vec[i][j] / norm_p;
        }
    }

    let mean_loss = total_loss / (b_size as f32);
    let loss_t = Tensor::from_vec(vec![mean_loss], 1, dev)?;

    let flat_norm: Vec<f32> = norm_pos_matrix.into_iter().flatten().collect();
    let h_norm_t = Tensor::from_vec(flat_norm, (b_size, d_size), dev)?;

    Ok((loss_t, h_norm_t))
}

/// Predictive Coding representation inference step (Rao & Ballard, 1999).
///
/// eps = x - r @ w
/// r_next = r + eta * (eps @ w^T - lam * r)
pub fn predictive_coding_step(
    x: &Tensor,
    r: &Tensor,
    w: &Tensor,
    eta: f64,
    lam: f64,
) -> Result<Tensor> {
    let r_row = r.unsqueeze(0)?; // [1, D]
    let pred = r_row.matmul(w)?.squeeze(0)?; // [D]
    let err = x.sub(&pred)?; // [D]
    let err_row = err.unsqueeze(0)?; // [1, D]

    let w_t = w.t()?;
    let grad = err_row.matmul(&w_t)?.squeeze(0)?; // [D]

    let step = (grad.sub(&(r * lam)?)? * eta)?;
    r.add(&step)
}

/// Direct Feedback Alignment (DFA) error projection (Nøkland, 2016).
///
/// err = (top_error @ dfa_feedback^T) * (act_pre > 0)
pub fn dfa_project_error(
    top_error: &Tensor,
    dfa_feedback: &Tensor,
    act_pre: &Tensor,
) -> Result<Tensor> {
    let (b_size, d_size) = act_pre.dims2()?;
    let dev = act_pre.device();

    let feedback_t = dfa_feedback.t()?; // [C, D]
    let proj = top_error.matmul(&feedback_t)?; // [B, D]

    let proj_vec = proj.to_vec2::<f32>()?;
    let act_vec = act_pre.to_vec2::<f32>()?;

    let mut out_vec = Vec::with_capacity(b_size * d_size);
    for i in 0..b_size {
        for j in 0..d_size {
            let mask = if act_vec[i][j] > 0.0f32 {
                1.0f32
            } else {
                0.0f32
            };
            out_vec.push(proj_vec[i][j] * mask);
        }
    }

    Tensor::from_vec(out_vec, (b_size, d_size), dev)
}

/// Oja's local unsupervised Hebbian PCA update (Oja, 1982).
///
/// y = x . w
/// w_next = w + eta * (y * x - y^2 * w)
pub fn oja_hebbian_step(x: &Tensor, w: &Tensor, eta: f64) -> Result<Tensor> {
    let y = (x * w)?.sum_all()?.to_scalar::<f32>()? as f64;
    let y_x = (x * y)?;
    let y2_w = (w * (y * y))?;
    let delta = (y_x.sub(&y2_w)? * eta)?;
    w.add(&delta)
}

/// Unified module encapsulating Non-Backprop algorithms.
pub struct NonBackpropUnifiedModule {
    pub h_pos: Tensor,
    pub h_neg: Tensor,
    pub golden_ff_loss: Tensor,
    pub golden_h_norm: Tensor,

    pub pc_x: Tensor,
    pub pc_r: Tensor,
    pub pc_w: Tensor,
    pub golden_pc_r_next: Tensor,

    pub top_error: Tensor,
    pub dfa_feedback: Tensor,
    pub act_pre: Tensor,
    pub golden_dfa_grad: Tensor,

    pub oja_x: Tensor,
    pub oja_w: Tensor,
    pub golden_oja_w_next: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl NonBackpropUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let h_pos = tensors.get("h_pos").unwrap().clone();
        let h_neg = tensors.get("h_neg").unwrap().clone();
        let golden_ff_loss = tensors.get("golden_ff_loss").unwrap().clone();
        let golden_h_norm = tensors.get("golden_h_norm").unwrap().clone();

        let pc_x = tensors.get("pc_x").unwrap().clone();
        let pc_r = tensors.get("pc_r").unwrap().clone();
        let pc_w = tensors.get("pc_w").unwrap().clone();
        let golden_pc_r_next = tensors.get("golden_pc_r_next").unwrap().clone();

        let top_error = tensors.get("top_error").unwrap().clone();
        let dfa_feedback = tensors.get("dfa_feedback").unwrap().clone();
        let act_pre = tensors.get("act_pre").unwrap().clone();
        let golden_dfa_grad = tensors.get("golden_dfa_grad").unwrap().clone();

        let oja_x = tensors.get("oja_x").unwrap().clone();
        let oja_w = tensors.get("oja_w").unwrap().clone();
        let golden_oja_w_next = tensors.get("golden_oja_w_next").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            h_pos,
            h_neg,
            golden_ff_loss,
            golden_h_norm,
            pc_x,
            pc_r,
            pc_w,
            golden_pc_r_next,
            top_error,
            dfa_feedback,
            act_pre,
            golden_dfa_grad,
            oja_x,
            oja_w,
            golden_oja_w_next,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. Forward-Forward invariant
        let (loss, h_norm) = forward_forward_loss_and_norm(&self.h_pos, &self.h_neg, 2.0, 1e-8)?;
        let err_loss = (&loss - &self.golden_ff_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_hn = (&h_norm - &self.golden_h_norm)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_ff = err_loss.max(err_hn);
        assert!(
            err_ff < atol,
            "Forward-Forward invariant failed: {err_ff:.2e} >= {atol:.2e}"
        );
        errs.insert("ff_err".to_string(), err_ff);

        // 2. Predictive Coding invariant
        let pc_next = predictive_coding_step(&self.pc_x, &self.pc_r, &self.pc_w, 0.1, 0.01)?;
        let err_pc = (&pc_next - &self.golden_pc_r_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_pc < atol,
            "Predictive Coding invariant failed: {err_pc:.2e} >= {atol:.2e}"
        );
        errs.insert("pc_err".to_string(), err_pc);

        // 3. DFA invariant
        let dfa_grad = dfa_project_error(&self.top_error, &self.dfa_feedback, &self.act_pre)?;
        let err_dfa = (&dfa_grad - &self.golden_dfa_grad)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_dfa < atol,
            "DFA invariant failed: {err_dfa:.2e} >= {atol:.2e}"
        );
        errs.insert("dfa_err".to_string(), err_dfa);

        // 4. Oja invariant
        let oja_next = oja_hebbian_step(&self.oja_x, &self.oja_w, 0.05)?;
        let err_oja = (&oja_next - &self.golden_oja_w_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_oja < atol,
            "Oja invariant failed: {err_oja:.2e} >= {atol:.2e}"
        );
        errs.insert("oja_err".to_string(), err_oja);

        // 5. Linear Anchor invariant
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = (&anchor_out - &self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_anchor < atol,
            "Anchor invariant failed: {err_anchor:.2e} >= {atol:.2e}"
        );
        errs.insert("anchor_err".to_string(), err_anchor);

        Ok(errs)
    }
}
