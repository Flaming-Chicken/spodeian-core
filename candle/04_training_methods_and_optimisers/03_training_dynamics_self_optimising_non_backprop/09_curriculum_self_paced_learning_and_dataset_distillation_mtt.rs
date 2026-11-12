//! Candle Implementation for Category 04 Topic 23 (Subfolder 03 Topic 09):
//! Curriculum, Self-Paced Learning, and Dataset Distillation (MTT).
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Self-Paced Learning (SPL) with hard and linear regularizers (Kumar et al., 2010).
//! 2. Competence-based automated curriculum scheduling (Platanios et al., 2019; Bengio et al., 2009).
//! 3. Dataset distillation via gradient matching (Zhao et al., 2021).
//! 4. Trajectory matching loss (MTT - Cazenavette et al., 2022).
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

/// Computes hard and linear Self-Paced Learning weights and losses (Kumar et al., 2010).
pub fn compute_spl_weights_and_loss(
    losses: &Tensor,
    pace_lambda: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dev = losses.device();
    let l_vec = losses.flatten_all()?.to_vec1::<f32>()?;
    let lambda_val = pace_lambda.flatten_all()?.to_vec1::<f32>()?[0];

    let n = l_vec.len();
    let mut h_vec = vec![0.0f32; n];
    let mut lin_vec = vec![0.0f32; n];

    let mut sum_h = 0.0f32;
    let mut sum_h_l = 0.0f32;
    let mut sum_lin = 0.0f32;
    let mut sum_lin_l = 0.0f32;

    for i in 0..n {
        let l = l_vec[i];
        if l < lambda_val {
            h_vec[i] = 1.0f32;
            sum_h += 1.0f32;
            sum_h_l += l;

            let w_lin = 1.0f32 - l / lambda_val;
            lin_vec[i] = w_lin;
            sum_lin += w_lin;
            sum_lin_l += w_lin * l;
        }
    }

    let weighted_loss_hard_val = sum_h_l / (sum_h + 1e-8f32);
    let weighted_loss_lin_val = sum_lin_l / (sum_lin + 1e-8f32);

    let h_t = Tensor::new(h_vec.as_slice(), dev)?;
    let lin_t = Tensor::new(lin_vec.as_slice(), dev)?;
    let l_h_t = Tensor::new(&[weighted_loss_hard_val], dev)?;
    let l_lin_t = Tensor::new(&[weighted_loss_lin_val], dev)?;

    Ok((h_t, lin_t, l_h_t, l_lin_t))
}

/// Computes competence function and active curriculum sample mask (Platanios et al., 2019).
pub fn compute_competence_and_curriculum_mask(
    difficulties: &Tensor,
    step_t: &Tensor,
    t_warmup: &Tensor,
    c0: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = difficulties.device();
    let diff_vec = difficulties.flatten_all()?.to_vec1::<f32>()?;
    let t_val = step_t.flatten_all()?.to_vec1::<f32>()?[0];
    let t_warm_val = t_warmup.flatten_all()?.to_vec1::<f32>()?[0];
    let c0_val = c0.flatten_all()?.to_vec1::<f32>()?[0];

    let comp_raw = c0_val + (1.0f32 - c0_val) * (t_val / t_warm_val);
    let comp_val = comp_raw.clamp(0.0f32, 1.0f32);

    let mut mask_vec = vec![0.0f32; diff_vec.len()];
    for i in 0..diff_vec.len() {
        if diff_vec[i] <= comp_val {
            mask_vec[i] = 1.0f32;
        }
    }

    let comp_t = Tensor::new(&[comp_val], dev)?;
    let mask_t = Tensor::new(mask_vec.as_slice(), dev)?;
    Ok((comp_t, mask_t))
}

/// Computes cosine distance gradient matching loss for dataset distillation (Zhao et al., 2021).
pub fn compute_gradient_matching_loss(grad_synth: &Tensor, grad_real: &Tensor) -> Result<Tensor> {
    let dev = grad_synth.device();
    let gs = grad_synth.flatten_all()?.to_vec1::<f32>()?;
    let gr = grad_real.flatten_all()?.to_vec1::<f32>()?;

    let mut dot = 0.0f32;
    let mut norm_s_sq = 0.0f32;
    let mut norm_r_sq = 0.0f32;

    for i in 0..gs.len() {
        dot += gs[i] * gr[i];
        norm_s_sq += gs[i] * gs[i];
        norm_r_sq += gr[i] * gr[i];
    }

    let norm_s = norm_s_sq.sqrt();
    let norm_r = norm_r_sq.sqrt();
    let cos_sim = dot / (norm_s * norm_r + 1e-8f32);
    let gm_loss = 1.0f32 - cos_sim;

    Tensor::new(&[gm_loss], dev)
}

/// Computes trajectory matching loss (MTT - Cazenavette et al., 2022).
pub fn compute_trajectory_matching_loss(
    theta_start: &Tensor,
    theta_student: &Tensor,
    theta_expert: &Tensor,
) -> Result<Tensor> {
    let dev = theta_start.device();
    let t_start = theta_start.flatten_all()?.to_vec1::<f32>()?;
    let t_stud = theta_student.flatten_all()?.to_vec1::<f32>()?;
    let t_exp = theta_expert.flatten_all()?.to_vec1::<f32>()?;

    let mut num = 0.0f32;
    let mut denom = 0.0f32;

    for i in 0..t_start.len() {
        let diff_stud = t_stud[i] - t_exp[i];
        let diff_start = t_start[i] - t_exp[i];
        num += diff_stud * diff_stud;
        denom += diff_start * diff_start;
    }

    let mtt_loss = num / (denom + 1e-8f32);
    Tensor::new(&[mtt_loss], dev)
}

/// Unified module for Category 04 Topic 23.
pub struct CurriculumDistillationUnifiedModule {
    pub losses: Tensor,
    pub pace_lambda: Tensor,
    pub golden_spl_hard_weights: Tensor,
    pub golden_spl_linear_weights: Tensor,
    pub golden_weighted_loss_hard: Tensor,
    pub golden_weighted_loss_linear: Tensor,

    pub difficulties: Tensor,
    pub step_t: Tensor,
    pub t_warmup: Tensor,
    pub c0: Tensor,
    pub golden_competence: Tensor,
    pub golden_active_curriculum_mask: Tensor,

    pub grad_real: Tensor,
    pub grad_synth: Tensor,
    pub golden_grad_matching_loss: Tensor,

    pub theta_start: Tensor,
    pub theta_student: Tensor,
    pub theta_expert: Tensor,
    pub golden_mtt_loss: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl CurriculumDistillationUnifiedModule {
    pub fn new(weights_path: &Path) -> Result<Self> {
        let dev = &Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, dev)?;

        let losses = tensors.get("losses").unwrap().clone();
        let pace_lambda = tensors.get("pace_lambda").unwrap().clone();
        let golden_spl_hard_weights = tensors.get("golden_spl_hard_weights").unwrap().clone();
        let golden_spl_linear_weights = tensors.get("golden_spl_linear_weights").unwrap().clone();
        let golden_weighted_loss_hard = tensors.get("golden_weighted_loss_hard").unwrap().clone();
        let golden_weighted_loss_linear =
            tensors.get("golden_weighted_loss_linear").unwrap().clone();

        let difficulties = tensors.get("difficulties").unwrap().clone();
        let step_t = tensors.get("step_t").unwrap().clone();
        let t_warmup = tensors.get("t_warmup").unwrap().clone();
        let c0 = tensors.get("c0").unwrap().clone();
        let golden_competence = tensors.get("golden_competence").unwrap().clone();
        let golden_active_curriculum_mask = tensors
            .get("golden_active_curriculum_mask")
            .unwrap()
            .clone();

        let grad_real = tensors.get("grad_real").unwrap().clone();
        let grad_synth = tensors.get("grad_synth").unwrap().clone();
        let golden_grad_matching_loss = tensors.get("golden_grad_matching_loss").unwrap().clone();

        let theta_start = tensors.get("theta_start").unwrap().clone();
        let theta_student = tensors.get("theta_student").unwrap().clone();
        let theta_expert = tensors.get("theta_expert").unwrap().clone();
        let golden_mtt_loss = tensors.get("golden_mtt_loss").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            losses,
            pace_lambda,
            golden_spl_hard_weights,
            golden_spl_linear_weights,
            golden_weighted_loss_hard,
            golden_weighted_loss_linear,
            difficulties,
            step_t,
            t_warmup,
            c0,
            golden_competence,
            golden_active_curriculum_mask,
            grad_real,
            grad_synth,
            golden_grad_matching_loss,
            theta_start,
            theta_student,
            theta_expert,
            golden_mtt_loss,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, tolerance: f32) -> Result<BTreeMap<String, f32>> {
        let mut errs = BTreeMap::new();

        let (h_w, l_w, l_h, l_l) = compute_spl_weights_and_loss(&self.losses, &self.pace_lambda)?;
        let diff_hw = h_w
            .flatten_all()?
            .sub(&self.golden_spl_hard_weights.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let diff_lw = l_w
            .flatten_all()?
            .sub(&self.golden_spl_linear_weights.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let diff_lh = (l_h.flatten_all()?.to_vec1::<f32>()?[0]
            - self
                .golden_weighted_loss_hard
                .flatten_all()?
                .to_vec1::<f32>()?[0])
            .abs();
        let diff_ll = (l_l.flatten_all()?.to_vec1::<f32>()?[0]
            - self
                .golden_weighted_loss_linear
                .flatten_all()?
                .to_vec1::<f32>()?[0])
            .abs();
        let spl_err = diff_hw.max(diff_lw).max(diff_lh).max(diff_ll);
        errs.insert("spl_err".to_string(), spl_err);

        let (comp, mask) = compute_competence_and_curriculum_mask(
            &self.difficulties,
            &self.step_t,
            &self.t_warmup,
            &self.c0,
        )?;
        let diff_comp = (comp.flatten_all()?.to_vec1::<f32>()?[0]
            - self.golden_competence.flatten_all()?.to_vec1::<f32>()?[0])
            .abs();
        let diff_mask = mask
            .flatten_all()?
            .sub(&self.golden_active_curriculum_mask.flatten_all()?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        let curriculum_err = diff_comp.max(diff_mask);
        errs.insert("curriculum_err".to_string(), curriculum_err);

        let gm_loss = compute_gradient_matching_loss(&self.grad_synth, &self.grad_real)?;
        let gm_err = (gm_loss.flatten_all()?.to_vec1::<f32>()?[0]
            - self
                .golden_grad_matching_loss
                .flatten_all()?
                .to_vec1::<f32>()?[0])
            .abs();
        errs.insert("gm_err".to_string(), gm_err);

        let mtt = compute_trajectory_matching_loss(
            &self.theta_start,
            &self.theta_student,
            &self.theta_expert,
        )?;
        let mtt_err = (mtt.flatten_all()?.to_vec1::<f32>()?[0]
            - self.golden_mtt_loss.flatten_all()?.to_vec1::<f32>()?[0])
            .abs();
        errs.insert("mtt_err".to_string(), mtt_err);

        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let diff_anchor = anchor_out
            .sub(&self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0.0f32, f32::max);
        errs.insert("anchor_err".to_string(), diff_anchor);

        for (k, v) in &errs {
            if *v >= tolerance {
                candle_core::bail!(
                    "Candle Topic 23 invariant {} error {} >= {}",
                    k,
                    v,
                    tolerance
                );
            }
        }

        Ok(errs)
    }
}
