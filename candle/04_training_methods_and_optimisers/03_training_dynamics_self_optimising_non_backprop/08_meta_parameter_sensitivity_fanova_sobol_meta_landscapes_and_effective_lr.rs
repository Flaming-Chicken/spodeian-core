//! Candle Implementation for Category 04 Topic 22 (Subfolder 03 Topic 08):
//! Meta-Parameter Sensitivity, fANOVA, Sobol' Indices, and Effective Learning Rate.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Functional ANOVA (fANOVA) variance decomposition and Sobol' indices (Sobol', 1993; Hutter et al., 2014).
//! 2. Scale-invariant Effective Learning Rate and equilibrium norm dynamics (Arora et al., 2019; Wan et al., 2021).
//! 3. Chaotic unrolled meta-gradient exponential explosion (Wu et al., 2018; Metz et al., 2019).
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

/// Helper for population variance (unbiased = false) of flat slices.
fn variance_slice(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let mut sum_sq = 0.0f32;
    for &v in values {
        let diff = v - mean;
        sum_sq += diff * diff;
    }
    sum_sq / n
}

/// Functional ANOVA (fANOVA) variance decomposition and Sobol' indices (Sobol', 1993; Hutter et al., 2014).
pub fn compute_fanova_sobol_indices(
    y_grid: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let dev = y_grid.device();
    let (m, n) = y_grid.dims2()?;
    let grid_data = y_grid.to_vec2::<f32>()?;

    let flat_grid: Vec<f32> = grid_data.iter().flatten().copied().collect();
    let total_var_val = variance_slice(&flat_grid);

    // E[Y | x1]: mean across columns (dim 1)
    let mut e_given_x1 = vec![0.0f32; m];
    for i in 0..m {
        e_given_x1[i] = grid_data[i].iter().sum::<f32>() / (n as f32);
    }
    let v1_val = variance_slice(&e_given_x1);

    // E[Y | x2]: mean across rows (dim 0)
    let mut e_given_x2 = vec![0.0f32; n];
    for j in 0..n {
        let mut sum_col = 0.0f32;
        for i in 0..m {
            sum_col += grid_data[i][j];
        }
        e_given_x2[j] = sum_col / (m as f32);
    }
    let v2_val = variance_slice(&e_given_x2);

    let v12_val = total_var_val - (v1_val + v2_val);

    let s1_val = v1_val / total_var_val;
    let s2_val = v2_val / total_var_val;
    let st1_val = (v1_val + v12_val) / total_var_val;
    let st2_val = (v2_val + v12_val) / total_var_val;

    let tot_t = Tensor::new(&[total_var_val], dev)?;
    let v1_t = Tensor::new(&[v1_val], dev)?;
    let v2_t = Tensor::new(&[v2_val], dev)?;
    let s1_t = Tensor::new(&[s1_val], dev)?;
    let s2_t = Tensor::new(&[s2_val], dev)?;
    let st1_t = Tensor::new(&[st1_val], dev)?;
    let st2_t = Tensor::new(&[st2_val], dev)?;

    Ok((tot_t, v1_t, v2_t, s1_t, s2_t, st1_t, st2_t))
}

/// Scale-invariant effective learning rate and equilibrium norm (Arora et al., 2019; Wan et al., 2021).
pub fn compute_effective_lr_and_eq_norm(
    w: &Tensor,
    eta_nominal: &Tensor,
    weight_decay: &Tensor,
    grad_norm: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = w.device();
    let w_vec = w.flatten_all()?.to_vec1::<f32>()?;
    let mut sum_sq = 0.0f32;
    for &v in &w_vec {
        sum_sq += v * v;
    }

    let eta = eta_nominal.flatten_all()?.to_vec1::<f32>()?[0];
    let wd = weight_decay.flatten_all()?.to_vec1::<f32>()?[0];
    let gn = grad_norm.flatten_all()?.to_vec1::<f32>()?[0];

    let eff_lr_val = eta / sum_sq;
    let eq_norm_val = ((eta * gn * gn) / (2.0f32 * wd)).powf(0.25f32);

    let eff_lr = Tensor::new(&[eff_lr_val], dev)?;
    let eq_norm = Tensor::new(&[eq_norm_val], dev)?;

    Ok((eff_lr, eq_norm))
}

/// Chaotic unrolled meta-gradient multiplier and exponential explosion (Metz et al., 2019).
pub fn compute_chaotic_meta_gradient_explosion(
    eig_max: &Tensor,
    eta_meta: &Tensor,
    steps_t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = eig_max.device();
    let em = eig_max.flatten_all()?.to_vec1::<f32>()?[0];
    let eta = eta_meta.flatten_all()?.to_vec1::<f32>()?[0];
    let t = steps_t.flatten_all()?.to_vec1::<f32>()?[0];

    let mult_val = (1.0f32 - eta * em).abs();
    let accum_val = mult_val.powf(t);

    let mult = Tensor::new(&[mult_val], dev)?;
    let accum = Tensor::new(&[accum_val], dev)?;

    Ok((mult, accum))
}

/// Unified Candle container for Category 04 Topic 22 verification.
pub struct MetaSensitivityUnifiedModule {
    pub y_grid: Tensor,
    pub golden_total_var: Tensor,
    pub golden_v1: Tensor,
    pub golden_v2: Tensor,
    pub golden_s1: Tensor,
    pub golden_s2: Tensor,
    pub golden_st1: Tensor,
    pub golden_st2: Tensor,

    pub w: Tensor,
    pub eta_nominal: Tensor,
    pub weight_decay: Tensor,
    pub grad_norm: Tensor,
    pub golden_effective_lr: Tensor,
    pub golden_eq_norm: Tensor,

    pub eig_max: Tensor,
    pub eta_meta: Tensor,
    pub steps_t: Tensor,
    pub golden_spectral_multiplier: Tensor,
    pub golden_accumulated_explosion: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl MetaSensitivityUnifiedModule {
    pub fn new(weights_path: &Path) -> Result<Self> {
        let dev = &Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, dev)?;

        let y_grid = tensors.get("y_grid").unwrap().clone();
        let golden_total_var = tensors.get("golden_total_var").unwrap().clone();
        let golden_v1 = tensors.get("golden_v1").unwrap().clone();
        let golden_v2 = tensors.get("golden_v2").unwrap().clone();
        let golden_s1 = tensors.get("golden_s1").unwrap().clone();
        let golden_s2 = tensors.get("golden_s2").unwrap().clone();
        let golden_st1 = tensors.get("golden_st1").unwrap().clone();
        let golden_st2 = tensors.get("golden_st2").unwrap().clone();

        let w = tensors.get("w").unwrap().clone();
        let eta_nominal = tensors.get("eta_nominal").unwrap().clone();
        let weight_decay = tensors.get("weight_decay").unwrap().clone();
        let grad_norm = tensors.get("grad_norm").unwrap().clone();
        let golden_effective_lr = tensors.get("golden_effective_lr").unwrap().clone();
        let golden_eq_norm = tensors.get("golden_eq_norm").unwrap().clone();

        let eig_max = tensors.get("eig_max").unwrap().clone();
        let eta_meta = tensors.get("eta_meta").unwrap().clone();
        let steps_t = tensors.get("steps_t").unwrap().clone();
        let golden_spectral_multiplier = tensors.get("golden_spectral_multiplier").unwrap().clone();
        let golden_accumulated_explosion =
            tensors.get("golden_accumulated_explosion").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            y_grid,
            golden_total_var,
            golden_v1,
            golden_v2,
            golden_s1,
            golden_s2,
            golden_st1,
            golden_st2,
            w,
            eta_nominal,
            weight_decay,
            grad_norm,
            golden_effective_lr,
            golden_eq_norm,
            eig_max,
            eta_meta,
            steps_t,
            golden_spectral_multiplier,
            golden_accumulated_explosion,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. Sobol invariant
        let (tot_var, v1, v2, s1, s2, st1, st2) = compute_fanova_sobol_indices(&self.y_grid)?;
        let err_tot = (&tot_var - &self.golden_total_var)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_v1 = (&v1 - &self.golden_v1)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_v2 = (&v2 - &self.golden_v2)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_s1 = (&s1 - &self.golden_s1)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_s2 = (&s2 - &self.golden_s2)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_st1 = (&st1 - &self.golden_st1)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_st2 = (&st2 - &self.golden_st2)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_sobol = err_tot
            .max(err_v1)
            .max(err_v2)
            .max(err_s1)
            .max(err_s2)
            .max(err_st1)
            .max(err_st2);
        assert!(
            err_sobol < atol,
            "Sobol invariant failed: {err_sobol:.2e} >= {atol:.2e}"
        );
        errs.insert("sobol_err".to_string(), err_sobol);

        // 2. Effective LR invariant
        let (eff_lr, eq_norm) = compute_effective_lr_and_eq_norm(
            &self.w,
            &self.eta_nominal,
            &self.weight_decay,
            &self.grad_norm,
        )?;
        let err_elr = (&eff_lr - &self.golden_effective_lr)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_eqn = (&eq_norm - &self.golden_eq_norm)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_eff = err_elr.max(err_eqn);
        assert!(
            err_eff < atol,
            "Effective LR invariant failed: {err_eff:.2e} >= {atol:.2e}"
        );
        errs.insert("eff_lr_err".to_string(), err_eff);

        // 3. Chaotic Meta-Gradient invariant
        let (mult, accum) =
            compute_chaotic_meta_gradient_explosion(&self.eig_max, &self.eta_meta, &self.steps_t)?;
        let err_mult = (&mult - &self.golden_spectral_multiplier)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_accum = (&accum - &self.golden_accumulated_explosion)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_chaos = err_mult.max(err_accum);
        assert!(
            err_chaos < atol,
            "Chaotic Meta-Gradient invariant failed: {err_chaos:.2e} >= {atol:.2e}"
        );
        errs.insert("chaos_err".to_string(), err_chaos);

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
