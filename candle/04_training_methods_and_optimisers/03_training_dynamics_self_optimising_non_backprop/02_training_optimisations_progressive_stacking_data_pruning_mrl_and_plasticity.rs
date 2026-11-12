//! Candle Implementation for Category 04 Topic 16 (Subfolder 03 Topic 02):
//! Training Optimisations: Progressive Stacking, Data Pruning, MRL, and Plasticity.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Matryoshka Representation Learning (MRL) multi-scale nested projection.
//! 2. EL2N (Error L2 Norm) early-training sample importance scoring.
//! 3. RHO-Loss (Reducible Holdout Loss) for unlearnable noise filtering.
//! 4. ReDo (Recycling Dormant Neurons) activation tracking and dormant mask identification.
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

/// Matryoshka Representation Learning (MRL) prefix projection (Kusupati et al., 2022).
///
/// Projects the first `dim_slice` dimensions: z[:, :dim_slice] @ w[:dim_slice, :] + b.
pub fn mrl_slice_forward(z: &Tensor, w: &Tensor, b: &Tensor, dim_slice: usize) -> Result<Tensor> {
    let z_sub = z.narrow(1, 0, dim_slice)?;
    let w_sub = w.narrow(0, 0, dim_slice)?;
    let xw = z_sub.matmul(&w_sub)?;
    xw.broadcast_add(b)
}

/// Error L2 Norm (EL2N) sample importance score (Paul et al., 2021).
///
/// Computes ||p(x) - y||_2 across class probabilities.
pub fn compute_el2n(probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = probs.sub(targets)?;
    let diff_sq = diff.sqr()?;
    let sum_sq = diff_sq.sum_keepdim(1)?;
    sum_sq.sqrt()?.flatten_all()
}

/// Reducible Holdout Loss (RHO-Loss) scoring (Mindermann et al., 2022).
///
/// Computes L_model - L_ref.
pub fn compute_rho_loss(loss_model: &Tensor, loss_ref: &Tensor) -> Result<Tensor> {
    loss_model.sub(loss_ref)
}

/// Detects dormant neurons via mean absolute activation threshold (Sokar et al., 2023).
///
/// Computes mean(|a_j|) along batch dimension and returns (mean_act, dormant_mask).
pub fn detect_dormant_neurons(activations: &Tensor, tau: f64) -> Result<(Tensor, Tensor)> {
    let abs_act = activations.abs()?;
    let mean_act = (abs_act.sum_keepdim(0)? * (1.0 / activations.dim(0)? as f64))?.flatten_all()?;
    let mean_vec = mean_act.to_vec1::<f32>()?;
    let t = tau as f32;

    let mask_vec: Vec<f32> = mean_vec
        .iter()
        .map(|&v| if v < t { 1.0f32 } else { 0.0f32 })
        .collect();
    let dormant_mask = Tensor::from_vec(mask_vec, mean_vec.len(), activations.device())?;

    Ok((mean_act, dormant_mask))
}

/// Unified module encapsulating MRL, Data Pruning, and Plasticity routines.
pub struct TrainingOptimisationsUnifiedModule {
    pub z_emb: Tensor,
    pub mrl_w: Tensor,
    pub mrl_b: Tensor,
    pub golden_logits_d2: Tensor,
    pub golden_logits_d4: Tensor,
    pub golden_logits_d8: Tensor,

    pub p_probs: Tensor,
    pub y_targets: Tensor,
    pub golden_el2n: Tensor,
    pub loss_model: Tensor,
    pub loss_ref: Tensor,
    pub golden_rho_loss: Tensor,

    pub activations: Tensor,
    pub golden_mean_act: Tensor,
    pub golden_dormant_mask: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl TrainingOptimisationsUnifiedModule {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let z_emb = tensors.get("z_emb").unwrap().clone();
        let mrl_w = tensors.get("mrl_w").unwrap().clone();
        let mrl_b = tensors.get("mrl_b").unwrap().clone();
        let golden_logits_d2 = tensors.get("golden_logits_d2").unwrap().clone();
        let golden_logits_d4 = tensors.get("golden_logits_d4").unwrap().clone();
        let golden_logits_d8 = tensors.get("golden_logits_d8").unwrap().clone();

        let p_probs = tensors.get("p_probs").unwrap().clone();
        let y_targets = tensors.get("y_targets").unwrap().clone();
        let golden_el2n = tensors.get("golden_el2n").unwrap().clone();
        let loss_model = tensors.get("loss_model").unwrap().clone();
        let loss_ref = tensors.get("loss_ref").unwrap().clone();
        let golden_rho_loss = tensors.get("golden_rho_loss").unwrap().clone();

        let activations = tensors.get("activations").unwrap().clone();
        let golden_mean_act = tensors.get("mean_act").unwrap().clone();
        let golden_dormant_mask = tensors.get("dormant_mask").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            z_emb,
            mrl_w,
            mrl_b,
            golden_logits_d2,
            golden_logits_d4,
            golden_logits_d8,
            p_probs,
            y_targets,
            golden_el2n,
            loss_model,
            loss_ref,
            golden_rho_loss,
            activations,
            golden_mean_act,
            golden_dormant_mask,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. MRL invariant
        let out_d2 = mrl_slice_forward(&self.z_emb, &self.mrl_w, &self.mrl_b, 2)?;
        let out_d4 = mrl_slice_forward(&self.z_emb, &self.mrl_w, &self.mrl_b, 4)?;
        let out_d8 = mrl_slice_forward(&self.z_emb, &self.mrl_w, &self.mrl_b, 8)?;

        let err_d2 = (&out_d2 - &self.golden_logits_d2)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_d4 = (&out_d4 - &self.golden_logits_d4)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_d8 = (&out_d8 - &self.golden_logits_d8)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_mrl = err_d2.max(err_d4).max(err_d8);
        assert!(
            err_mrl < atol,
            "MRL invariant failed: {err_mrl:.2e} >= {atol:.2e}"
        );
        errs.insert("mrl_err".to_string(), err_mrl);

        // 2. EL2N invariant
        let el2n = compute_el2n(&self.p_probs, &self.y_targets)?;
        let err_el2n = (&el2n - &self.golden_el2n)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_el2n < atol,
            "EL2N invariant failed: {err_el2n:.2e} >= {atol:.2e}"
        );
        errs.insert("el2n_err".to_string(), err_el2n);

        // 3. RHO-Loss invariant
        let rho = compute_rho_loss(&self.loss_model, &self.loss_ref)?;
        let err_rho = (&rho - &self.golden_rho_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_rho < atol,
            "RHO-loss invariant failed: {err_rho:.2e} >= {atol:.2e}"
        );
        errs.insert("rho_loss_err".to_string(), err_rho);

        // 4. ReDo invariant
        let (mean_act, mask) = detect_dormant_neurons(&self.activations, 0.15)?;
        let err_mean = (&mean_act - &self.golden_mean_act)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_mask = (&mask - &self.golden_dormant_mask)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_redo = err_mean.max(err_mask);
        assert!(
            err_redo < atol,
            "ReDo invariant failed: {err_redo:.2e} >= {atol:.2e}"
        );
        errs.insert("redo_err".to_string(), err_redo);

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
