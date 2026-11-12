//! Candle Implementation for Category 04 Topic 17 (Subfolder 03 Topic 03):
//! Self-Optimising Hypergradients, Learned Optimisers, and Meta-Reweighting.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Hypergradient Descent (HD) online learning rate adaptation (Baydin et al., 2018).
//! 2. Meta-Sample Reweighting for noisy label robustness (Ren et al., 2018).
//! 3. Coordinate-wise Learned Optimizer MLP step (Andrychowicz et al., 2016).
//! 4. Linear anchor evaluation for transitive multi-track parity.

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

/// Hypergradient Descent (HD) step (Baydin et al., 2018).
///
/// Computes:
/// h_t = <g_curr, g_prev>
/// lr_{t+1} = lr_t + hyper_lr * h_t
/// w_{t+1} = w_t - lr_{t+1} * g_curr
pub fn step_hypergradient_descent(
    w: &Tensor,
    g_curr: &Tensor,
    g_prev: &Tensor,
    lr: &Tensor,
    hyper_lr: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let h_t = (g_curr * g_prev)?.sum_all()?.to_scalar::<f32>()? as f64;
    let lr_delta = (hyper_lr * h_t)?;
    let lr_next = lr.add(&lr_delta)?;
    let lr_val = lr_next.flatten_all()?.to_vec1::<f32>()?[0] as f64;
    let step = (g_curr * lr_val)?;
    let w_next = w.sub(&step)?;
    Ok((w_next, lr_next))
}

/// Meta-Sample Reweighting (Ren et al., 2018).
///
/// Computes sample alignment with validation gradient:
/// s_i = <G_i, val_grad>
/// v_i = max(0, s_i)
/// norm_v = v / (sum(v) + eps)
/// reweighted_grad = norm_v @ G
pub fn meta_sample_reweight(
    per_sample_grads: &Tensor,
    val_grad: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    // per_sample_grads: [B, D], val_grad: [D]
    let val_grad_col = val_grad.unsqueeze(1)?; // [D, 1]
    let alignments = per_sample_grads.matmul(&val_grad_col)?.squeeze(1)?; // [B]

    let align_vec = alignments.to_vec1::<f32>()?;
    let mut relu_vec = Vec::with_capacity(align_vec.len());
    let mut sum_v = 0.0f32;
    for &a in &align_vec {
        let v = if a > 0.0 { a } else { 0.0 };
        relu_vec.push(v);
        sum_v += v;
    }
    let denom = sum_v + eps as f32;
    let norm_vec: Vec<f32> = relu_vec.iter().map(|&v| v / denom).collect();

    let norm_weights = Tensor::from_vec(norm_vec, align_vec.len(), per_sample_grads.device())?;
    let weights_row = norm_weights.unsqueeze(0)?; // [1, B]
    let reweighted_grad = weights_row.matmul(per_sample_grads)?.squeeze(0)?; // [D]

    Ok((norm_weights, reweighted_grad))
}

/// Coordinate-wise Learned Optimizer MLP step (Andrychowicz et al., 2016).
///
/// Input coords: [D, 2] stacked (grad, momentum)
/// h = relu(coords @ w1 + b1)  [D, 4]
/// delta_w = h @ w2 + b2       [D]
/// p_next = param - delta_w
/// m_next = 0.9 * momentum + 0.1 * grad
pub fn step_learned_optimizer(
    param: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let d = grad.dim(0)?;
    let g_col = grad.unsqueeze(1)?;
    let m_col = momentum.unsqueeze(1)?;
    let coords = Tensor::cat(&[&g_col, &m_col], 1)?; // [D, 2]

    let h = coords.matmul(w1)?.broadcast_add(b1)?.relu()?; // [D, 4]
    let delta_w = h.matmul(w2)?.broadcast_add(b2)?.reshape(&[d])?; // [D]

    let p_next = param.sub(&delta_w)?;
    let m_next = ((momentum * 0.9)? + (grad * 0.1)?)?;

    Ok((p_next, m_next, delta_w))
}

/// Unified module encapsulating Hypergradients, Learned Optimizers, and Meta-Reweighting.
pub struct HypergradientsUnifiedModule {
    pub w_hd: Tensor,
    pub g_curr: Tensor,
    pub g_prev: Tensor,
    pub lr_init: Tensor,
    pub hyper_lr: Tensor,
    pub golden_lr_next: Tensor,
    pub golden_w_hd: Tensor,

    pub per_sample_grads: Tensor,
    pub val_grad: Tensor,
    pub golden_sample_weights: Tensor,
    pub golden_reweighted_grad: Tensor,

    pub param_learned: Tensor,
    pub grad_learned: Tensor,
    pub momentum_learned: Tensor,
    pub mlp_w1: Tensor,
    pub mlp_b1: Tensor,
    pub mlp_w2: Tensor,
    pub mlp_b2: Tensor,
    pub golden_delta_w: Tensor,
    pub golden_param_learned: Tensor,
    pub golden_momentum_learned: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl HypergradientsUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let w_hd = tensors.get("w_hd").unwrap().clone();
        let g_curr = tensors.get("g_curr").unwrap().clone();
        let g_prev = tensors.get("g_prev").unwrap().clone();
        let lr_init = tensors.get("lr_init").unwrap().clone();
        let hyper_lr = tensors.get("hyper_lr").unwrap().clone();
        let golden_lr_next = tensors.get("golden_lr_next").unwrap().clone();
        let golden_w_hd = tensors.get("golden_w_hd").unwrap().clone();

        let per_sample_grads = tensors.get("per_sample_grads").unwrap().clone();
        let val_grad = tensors.get("val_grad").unwrap().clone();
        let golden_sample_weights = tensors.get("golden_sample_weights").unwrap().clone();
        let golden_reweighted_grad = tensors.get("golden_reweighted_grad").unwrap().clone();

        let param_learned = tensors.get("param_learned").unwrap().clone();
        let grad_learned = tensors.get("grad_learned").unwrap().clone();
        let momentum_learned = tensors.get("momentum_learned").unwrap().clone();
        let mlp_w1 = tensors.get("mlp_w1").unwrap().clone();
        let mlp_b1 = tensors.get("mlp_b1").unwrap().clone();
        let mlp_w2 = tensors.get("mlp_w2").unwrap().clone();
        let mlp_b2 = tensors.get("mlp_b2").unwrap().clone();
        let golden_delta_w = tensors.get("golden_delta_w").unwrap().clone();
        let golden_param_learned = tensors.get("golden_param_learned").unwrap().clone();
        let golden_momentum_learned = tensors.get("golden_momentum_learned").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            w_hd,
            g_curr,
            g_prev,
            lr_init,
            hyper_lr,
            golden_lr_next,
            golden_w_hd,
            per_sample_grads,
            val_grad,
            golden_sample_weights,
            golden_reweighted_grad,
            param_learned,
            grad_learned,
            momentum_learned,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            golden_delta_w,
            golden_param_learned,
            golden_momentum_learned,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. Hypergradient Descent invariant
        let (w_out, lr_out) = step_hypergradient_descent(
            &self.w_hd,
            &self.g_curr,
            &self.g_prev,
            &self.lr_init,
            &self.hyper_lr,
        )?;
        let err_lr = (&lr_out - &self.golden_lr_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_w = (&w_out - &self.golden_w_hd)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_hd = err_lr.max(err_w);
        assert!(
            err_hd < atol,
            "HD invariant failed: {err_hd:.2e} >= {atol:.2e}"
        );
        errs.insert("hd_err".to_string(), err_hd);

        // 2. Meta-Sample Reweighting invariant
        let (weights_out, rew_g_out) =
            meta_sample_reweight(&self.per_sample_grads, &self.val_grad, 1e-8)?;
        let err_weights = (&weights_out - &self.golden_sample_weights)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_rew_g = (&rew_g_out - &self.golden_reweighted_grad)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_reweight = err_weights.max(err_rew_g);
        assert!(
            err_reweight < atol,
            "Reweight invariant failed: {err_reweight:.2e} >= {atol:.2e}"
        );
        errs.insert("reweight_err".to_string(), err_reweight);

        // 3. Learned Optimizer invariant
        let (p_out, m_out, d_w_out) = step_learned_optimizer(
            &self.param_learned,
            &self.grad_learned,
            &self.momentum_learned,
            &self.mlp_w1,
            &self.mlp_b1,
            &self.mlp_w2,
            &self.mlp_b2,
        )?;
        let err_p = (&p_out - &self.golden_param_learned)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_m = (&m_out - &self.golden_momentum_learned)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_d = (&d_w_out - &self.golden_delta_w)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_learned = err_p.max(err_m).max(err_d);
        assert!(
            err_learned < atol,
            "Learned optimizer invariant failed: {err_learned:.2e} >= {atol:.2e}"
        );
        errs.insert("learned_err".to_string(), err_learned);

        // 4. Linear Anchor invariant
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
