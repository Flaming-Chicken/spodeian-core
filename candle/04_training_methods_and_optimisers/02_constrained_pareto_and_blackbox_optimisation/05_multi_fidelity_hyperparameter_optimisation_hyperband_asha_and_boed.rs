//! Candle Implementation for Category 04 Topic 12 (Subfolder 02 Topic 05):
//! Multi-Fidelity Hyperparameter Optimisation: Hyperband, ASHA, and BOED.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Successive Halving Algorithm (SHA) multi-fidelity configuration pruning.
//! 2. Hyperband bracket resource allocation schedule calculation.
//! 3. Tree-structured Parzen Estimator (TPE) non-parametric density ratio surrogate.
//! 4. Asynchronous Successive Halving (ASHA) non-blocking promotion logic.

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

/// Select top k = n // eta candidates with lowest loss.
pub fn compute_sha_top_k(losses: &Tensor, eta: usize) -> Result<(Tensor, Tensor)> {
    let loss_vec = losses.to_vec1::<f32>()?;
    let n = loss_vec.len();
    let k = (n / eta).max(1);

    let mut indexed: Vec<(usize, f32)> = loss_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out_idx = vec![0.0f32; k];
    let mut out_loss = vec![0.0f32; k];
    for i in 0..k {
        out_idx[i] = indexed[i].0 as f32;
        out_loss[i] = indexed[i].1;
    }

    let dev = losses.device();
    Ok((
        Tensor::from_vec(out_idx, k, dev)?,
        Tensor::from_vec(out_loss, k, dev)?,
    ))
}

/// Compute (n_s, r_s) pairs across brackets s in [s_max, ..., 0].
pub fn compute_hyperband_schedule(r_max: usize, eta: usize, dev: &Device) -> Result<Tensor> {
    let s_max = ((r_max as f64).ln() / (eta as f64).ln()).floor() as usize;
    let b = ((s_max + 1) * r_max) as f64;
    let num_brackets = s_max + 1;
    let mut schedule = vec![0.0f32; num_brackets * 2];

    for (row, s) in (0..=s_max).rev().enumerate() {
        let eta_s = (eta as f64).powi(s as i32);
        let n_s = ((b / (r_max as f64)) * (eta_s / ((s + 1) as f64))).ceil() as f32;
        let r_s = ((r_max as f64) * (eta as f64).powi(-(s as i32))) as f32;
        schedule[row * 2] = n_s;
        schedule[row * 2 + 1] = r_s;
    }

    Tensor::from_vec(schedule, (num_brackets, 2), dev)
}

/// Evaluate 1D TPE log-likelihood ratio: log l(x) - log g(x).
pub fn compute_tpe_log_ratio(
    x_query: &Tensor,
    good_samples: &Tensor,
    bad_samples: &Tensor,
    bandwidth: f64,
) -> Result<Tensor> {
    let q_vec = x_query.to_vec1::<f32>()?;
    let good_vec = good_samples.to_vec1::<f32>()?;
    let bad_vec = bad_samples.to_vec1::<f32>()?;

    let log_kde = |q: &[f32], s: &[f32]| -> Vec<f64> {
        let pi = std::f64::consts::PI;
        let n = s.len() as f64;
        let mut out = vec![0.0f64; q.len()];
        for i in 0..q.len() {
            let x = q[i] as f64;
            // logsumexp over samples
            let mut log_terms = vec![0.0f64; s.len()];
            let mut max_term = -1e30f64;
            for j in 0..s.len() {
                let diff = (x - (s[j] as f64)) / bandwidth;
                let term = -0.5 * diff * diff - 0.5 * (2.0 * pi).ln() - bandwidth.ln();
                log_terms[j] = term;
                if term > max_term {
                    max_term = term;
                }
            }
            let mut sum_exp = 0.0f64;
            for j in 0..s.len() {
                sum_exp += (log_terms[j] - max_term).exp();
            }
            out[i] = max_term + sum_exp.ln() - n.ln();
        }
        out
    };

    let log_l = log_kde(&q_vec, &good_vec);
    let log_g = log_kde(&q_vec, &bad_vec);

    let mut ratios = vec![0.0f32; q_vec.len()];
    for i in 0..q_vec.len() {
        ratios[i] = (log_l[i] - log_g[i]) as f32;
    }

    Tensor::from_vec(ratios, q_vec.len(), x_query.device())
}

/// Return 1.0 if candidate_loss is eligible for promotion (in top 1/eta), else 0.0.
pub fn check_asha_promotion(rung_losses: &Tensor, candidate_loss: f64, eta: usize) -> Result<f64> {
    let mut loss_vec = rung_losses.to_vec1::<f32>()?;
    let k = (loss_vec.len() / eta).max(1);
    loss_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = loss_vec[k - 1] as f64;
    Ok(if candidate_loss <= cutoff { 1.0 } else { 0.0 })
}

/// Unified module encapsulating multi-fidelity HPO algorithms for parity checks.
pub struct MultiFidelityUnifiedModule {
    pub sha_losses: Tensor,
    pub golden_sha_idx: Tensor,
    pub golden_sha_losses: Tensor,
    pub golden_hb_schedule: Tensor,
    pub x_query: Tensor,
    pub good_samples: Tensor,
    pub bad_samples: Tensor,
    pub golden_tpe_log_ratio: Tensor,
    pub rung_losses: Tensor,
    pub candidate_losses: Tensor,
    pub golden_asha_promoted: Tensor,
    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl MultiFidelityUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let sha_losses = tensors
            .get("sha_losses")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sha_losses".into()))?;
        let golden_sha_idx = tensors
            .get("golden_sha_idx")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sha_idx".into()))?;
        let golden_sha_losses = tensors
            .get("golden_sha_losses")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sha_losses".into()))?;
        let golden_hb_schedule = tensors
            .get("golden_hb_schedule")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_hb_schedule".into()))?;

        let x_query = tensors
            .get("x_query")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing x_query".into()))?;
        let good_samples = tensors
            .get("good_samples")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing good_samples".into()))?;
        let bad_samples = tensors
            .get("bad_samples")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing bad_samples".into()))?;
        let golden_tpe_log_ratio = tensors
            .get("golden_tpe_log_ratio")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_tpe_log_ratio".into()))?;

        let rung_losses = tensors
            .get("rung_losses")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing rung_losses".into()))?;
        let candidate_losses = tensors
            .get("candidate_losses")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing candidate_losses".into()))?;
        let golden_asha_promoted = tensors
            .get("golden_asha_promoted")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_asha_promoted".into()))?;

        let anchor_w = tensors
            .get("anchor_w")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_w".into()))?;
        let anchor_b = tensors
            .get("anchor_b")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_b".into()))?;
        let anchor = LinearAnchor::new(anchor_w, anchor_b);
        let anchor_x = tensors
            .get("anchor_x")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing anchor_x".into()))?;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;

        Ok(Self {
            sha_losses,
            golden_sha_idx,
            golden_sha_losses,
            golden_hb_schedule,
            x_query,
            good_samples,
            bad_samples,
            golden_tpe_log_ratio,
            rung_losses,
            candidate_losses,
            golden_asha_promoted,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // 1. SHA Invariant
        let (idx, losses) = compute_sha_top_k(&self.sha_losses, 3)?;
        let err_idx = max_abs_diff(&idx, &self.golden_sha_idx)?;
        let err_l = max_abs_diff(&losses, &self.golden_sha_losses)?;
        let err_sha = err_idx.max(err_l);
        if err_sha >= atol {
            candle_core::bail!("SHA invariant failed: err={:e} >= {:e}", err_sha, atol);
        }
        errors.insert("sha_err".to_string(), err_sha);

        // 2. Hyperband Schedule Invariant
        let schedule = compute_hyperband_schedule(81, 3, self.sha_losses.device())?;
        let err_hb = max_abs_diff(&schedule, &self.golden_hb_schedule)?;
        if err_hb >= atol {
            candle_core::bail!("Hyperband invariant failed: err={:e} >= {:e}", err_hb, atol);
        }
        errors.insert("hyperband_err".to_string(), err_hb);

        // 3. TPE Log Ratio Invariant
        let tpe_ratio =
            compute_tpe_log_ratio(&self.x_query, &self.good_samples, &self.bad_samples, 0.5)?;
        let err_tpe = max_abs_diff(&tpe_ratio, &self.golden_tpe_log_ratio)?;
        if err_tpe >= atol {
            candle_core::bail!("TPE invariant failed: err={:e} >= {:e}", err_tpe, atol);
        }
        errors.insert("tpe_err".to_string(), err_tpe);

        // 4. ASHA Invariant
        let cand_vec = self.candidate_losses.to_vec1::<f32>()?;
        let p0 = check_asha_promotion(&self.rung_losses, cand_vec[0] as f64, 3)?;
        let p1 = check_asha_promotion(&self.rung_losses, cand_vec[1] as f64, 3)?;
        let asha_promoted =
            Tensor::from_vec(vec![p0 as f32, p1 as f32], 2, self.sha_losses.device())?;
        let err_asha = max_abs_diff(&asha_promoted, &self.golden_asha_promoted)?;
        if err_asha >= atol {
            candle_core::bail!("ASHA invariant failed: err={:e} >= {:e}", err_asha, atol);
        }
        errors.insert("asha_err".to_string(), err_asha);

        // 5. Linear Anchor Invariant
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = max_abs_diff(&anchor_out, &self.golden_anchor_out)?;
        if err_anchor >= atol {
            candle_core::bail!(
                "Anchor invariant failed: err={:e} >= {:e}",
                err_anchor,
                atol
            );
        }
        errors.insert("anchor_err".to_string(), err_anchor);

        Ok(errors)
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
    let diff = a.sub(b)?.abs()?;
    let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(max as f64)
}
