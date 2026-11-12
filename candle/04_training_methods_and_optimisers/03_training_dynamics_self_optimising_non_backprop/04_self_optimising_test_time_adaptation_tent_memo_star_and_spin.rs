//! Candle Implementation for Category 04 Topic 18 (Subfolder 03 Topic 04):
//! Self-Optimising Test-Time Adaptation: TENT, MEMO, STaR, and SPIN.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. TENT: Test-Time Entropy Minimization updating normalization affine parameters (Wang et al., 2021).
//! 2. MEMO: Marginal Entropy Minimization with One test sample across augmentations (Zhang et al., 2022).
//! 3. SPIN: Self-Play Fine-Tuning preference optimization (Chen et al., 2024).
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

/// TENT step updating normalization scale and shift via analytical entropy gradient.
///
/// Closed-form entropy derivative w.r.t scaled logits:
/// dH/dz_c = - p_c * (log(p_c) + H)
/// dL/d_scale_c = 1/B sum_i (dH/dz_{i,c} * logits_{i,c})
/// dL/d_shift_c = 1/B sum_i dH/dz_{i,c}
pub fn compute_tent_step(
    logits: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    lr: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (b_size, c_size) = logits.dims2()?;
    let dev = logits.device();

    let scaled = (logits.broadcast_mul(scale)?.broadcast_add(shift))?;
    let probs = candle_nn::ops::softmax(&scaled, 1)?;
    let log_probs = candle_nn::ops::log_softmax(&scaled, 1)?;

    let p_vec = probs.to_vec2::<f32>()?;
    let log_p_vec = log_probs.to_vec2::<f32>()?;
    let l_vec = logits.to_vec2::<f32>()?;

    let mut sample_entropies = vec![0.0f32; b_size];
    let mut total_entropy = 0.0f32;

    for i in 0..b_size {
        let mut h_i = 0.0f32;
        for c in 0..c_size {
            h_i -= p_vec[i][c] * log_p_vec[i][c];
        }
        sample_entropies[i] = h_i;
        total_entropy += h_i;
    }
    let mean_entropy = total_entropy / (b_size as f32);

    let mut grad_scale = vec![0.0f32; c_size];
    let mut grad_shift = vec![0.0f32; c_size];

    let inv_b = 1.0f32 / (b_size as f32);
    for i in 0..b_size {
        let h_i = sample_entropies[i];
        for c in 0..c_size {
            let p = p_vec[i][c];
            let lp = log_p_vec[i][c];
            let dh_dz = -p * (lp + h_i);
            grad_scale[c] += inv_b * dh_dz * l_vec[i][c];
            grad_shift[c] += inv_b * dh_dz;
        }
    }

    let g_scale_t = Tensor::from_vec(grad_scale.clone(), c_size, dev)?;
    let g_shift_t = Tensor::from_vec(grad_shift.clone(), c_size, dev)?;

    let scale_vec = scale.to_vec1::<f32>()?;
    let shift_vec = shift.to_vec1::<f32>()?;

    let lr_f = lr as f32;
    let scale_next_vec: Vec<f32> = (0..c_size)
        .map(|c| scale_vec[c] - lr_f * grad_scale[c])
        .collect();
    let shift_next_vec: Vec<f32> = (0..c_size)
        .map(|c| shift_vec[c] - lr_f * grad_shift[c])
        .collect();

    let scale_next = Tensor::from_vec(scale_next_vec, c_size, dev)?;
    let shift_next = Tensor::from_vec(shift_next_vec, c_size, dev)?;
    let entropy_t = Tensor::from_vec(vec![mean_entropy], 1, dev)?;

    Ok((scale_next, shift_next, entropy_t, g_scale_t, g_shift_t))
}

/// MEMO marginal entropy minimization across K test augmentations of a single sample (Zhang et al., 2022).
pub fn compute_memo_marginal_entropy(aug_logits: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
    let (k_size, c_size) = aug_logits.dims2()?;
    let dev = aug_logits.device();
    let probs = candle_nn::ops::softmax(aug_logits, 1)?;
    let p_vec = probs.to_vec2::<f32>()?;

    let inv_k = 1.0f32 / (k_size as f32);
    let mut marginal_p = vec![0.0f32; c_size];
    for k in 0..k_size {
        for c in 0..c_size {
            marginal_p[c] += p_vec[k][c] * inv_k;
        }
    }

    let mut entropy = 0.0f32;
    let eps_f = eps as f32;
    for c in 0..c_size {
        let p = marginal_p[c];
        entropy -= p * (p + eps_f).ln();
    }

    let marg_t = Tensor::from_vec(marginal_p, c_size, dev)?;
    let ent_t = Tensor::from_vec(vec![entropy], 1, dev)?;
    Ok((marg_t, ent_t))
}

/// SPIN / STaR self-play preference loss (Chen et al., 2024).
///
/// loss = - mean(log_sigmoid(beta * (logp_win - logp_lose)))
pub fn compute_spin_loss(logp_win: &Tensor, logp_lose: &Tensor, beta: f64) -> Result<Tensor> {
    let diff = (logp_win.sub(logp_lose)? * beta)?;
    let diff_vec = diff.to_vec1::<f32>()?;

    let mut sum_loss = 0.0f32;
    for &d in &diff_vec {
        // log_sigmoid(x) = -ln(1 + exp(-x)) = -softplus(-x)
        let log_sig = -((-d).exp().ln_1p());
        sum_loss -= log_sig;
    }
    let mean_loss = sum_loss / (diff_vec.len() as f32);
    Tensor::from_vec(vec![mean_loss], 1, logp_win.device())
}

/// Unified module encapsulating TENT, MEMO, and SPIN test-time adaptation.
pub struct TestTimeAdaptationUnifiedModule {
    pub tent_logits: Tensor,
    pub tent_scale: Tensor,
    pub tent_shift: Tensor,
    pub golden_tent_entropy: Tensor,
    pub golden_scale_grad: Tensor,
    pub golden_shift_grad: Tensor,
    pub golden_scale_next: Tensor,
    pub golden_shift_next: Tensor,

    pub memo_aug_logits: Tensor,
    pub golden_marginal_p: Tensor,
    pub golden_memo_entropy: Tensor,

    pub logp_win: Tensor,
    pub logp_lose: Tensor,
    pub golden_spin_loss: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl TestTimeAdaptationUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let tent_logits = tensors.get("tent_logits").unwrap().clone();
        let tent_scale = tensors.get("tent_scale").unwrap().clone();
        let tent_shift = tensors.get("tent_shift").unwrap().clone();
        let golden_tent_entropy = tensors.get("golden_tent_entropy").unwrap().clone();
        let golden_scale_grad = tensors.get("golden_scale_grad").unwrap().clone();
        let golden_shift_grad = tensors.get("golden_shift_grad").unwrap().clone();
        let golden_scale_next = tensors.get("golden_scale_next").unwrap().clone();
        let golden_shift_next = tensors.get("golden_shift_next").unwrap().clone();

        let memo_aug_logits = tensors.get("memo_aug_logits").unwrap().clone();
        let golden_marginal_p = tensors.get("golden_marginal_p").unwrap().clone();
        let golden_memo_entropy = tensors.get("golden_memo_entropy").unwrap().clone();

        let logp_win = tensors.get("logp_win").unwrap().clone();
        let logp_lose = tensors.get("logp_lose").unwrap().clone();
        let golden_spin_loss = tensors.get("golden_spin_loss").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            tent_logits,
            tent_scale,
            tent_shift,
            golden_tent_entropy,
            golden_scale_grad,
            golden_shift_grad,
            golden_scale_next,
            golden_shift_next,
            memo_aug_logits,
            golden_marginal_p,
            golden_memo_entropy,
            logp_win,
            logp_lose,
            golden_spin_loss,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. TENT invariant
        let (s_next, sh_next, ent, g_s, g_sh) =
            compute_tent_step(&self.tent_logits, &self.tent_scale, &self.tent_shift, 0.05)?;
        let err_ent = (&ent - &self.golden_tent_entropy)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_s = (&s_next - &self.golden_scale_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_sh = (&sh_next - &self.golden_shift_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_gs = (&g_s - &self.golden_scale_grad)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_gsh = (&g_sh - &self.golden_shift_grad)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_tent = err_ent.max(err_s).max(err_sh).max(err_gs).max(err_gsh);
        assert!(
            err_tent < atol,
            "TENT invariant failed: {err_tent:.2e} >= {atol:.2e}"
        );
        errs.insert("tent_err".to_string(), err_tent);

        // 2. MEMO invariant
        let (marg_p, memo_ent) = compute_memo_marginal_entropy(&self.memo_aug_logits, 1e-8)?;
        let err_p = (&marg_p - &self.golden_marginal_p)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_me = (&memo_ent - &self.golden_memo_entropy)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_memo = err_p.max(err_me);
        assert!(
            err_memo < atol,
            "MEMO invariant failed: {err_memo:.2e} >= {atol:.2e}"
        );
        errs.insert("memo_err".to_string(), err_memo);

        // 3. SPIN invariant
        let spin_loss = compute_spin_loss(&self.logp_win, &self.logp_lose, 0.5)?;
        let err_spin = (&spin_loss - &self.golden_spin_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_spin < atol,
            "SPIN invariant failed: {err_spin:.2e} >= {atol:.2e}"
        );
        errs.insert("spin_err".to_string(), err_spin);

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
