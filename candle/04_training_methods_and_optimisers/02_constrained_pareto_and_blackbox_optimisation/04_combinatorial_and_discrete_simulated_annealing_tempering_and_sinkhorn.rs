//! Candle Implementation for Category 04 Topic 11 (Subfolder 02 Topic 04):
//! Combinatorial and Discrete: Simulated Annealing, Parallel Tempering, and Sinkhorn.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. Gumbel-Sinkhorn continuous permutation relaxation via alternating row/column scalings.
//! 2. Cross-Entropy Method (CEM) distribution parameter estimation.
//! 3. Metropolis-Hastings acceptance probability for Simulated Annealing (SA).
//! 4. Replica-Exchange Parallel Tempering swap probability.

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

/// Sinkhorn normalization over square cost/score matrix.
pub fn run_sinkhorn(log_alpha: &Tensor, n_iters: usize, tau: f64) -> Result<Tensor> {
    let p_init = (log_alpha / tau)?.exp()?;
    let mut p = p_init;
    let eps = 1e-12;

    for _ in 0..n_iters {
        // Row normalization: P / (sum_cols + eps)
        let row_sum = p.sum_keepdim(1)?;
        let row_sum_eps = (row_sum + eps)?;
        p = p.broadcast_div(&row_sum_eps)?;

        // Column normalization: P / (sum_rows + eps)
        let col_sum = p.sum_keepdim(0)?;
        let col_sum_eps = (col_sum + eps)?;
        p = p.broadcast_div(&col_sum_eps)?;
    }

    Ok(p)
}

/// Cross-Entropy Method step: select elite candidates, update mean and std.
pub fn step_cem(candidates: &Tensor, scores: &Tensor, elite_frac: f64) -> Result<(Tensor, Tensor)> {
    let cand_vec = candidates.to_vec2::<f32>()?;
    let scores_vec = scores.to_vec1::<f32>()?;
    let n = cand_vec.len();
    let d = cand_vec[0].len();

    let mut indexed_scores: Vec<(usize, f32)> = scores_vec.into_iter().enumerate().collect();
    indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_elite = ((n as f64) * elite_frac).floor() as usize;
    let n_elite = n_elite.max(1);

    let mut mean = vec![0.0f32; d];
    for &(idx, _) in indexed_scores.iter().take(n_elite) {
        for j in 0..d {
            mean[j] += cand_vec[idx][j];
        }
    }
    for j in 0..d {
        mean[j] /= n_elite as f32;
    }

    let mut std = vec![0.0f32; d];
    for &(idx, _) in indexed_scores.iter().take(n_elite) {
        for j in 0..d {
            let diff = cand_vec[idx][j] - mean[j];
            std[j] += diff * diff;
        }
    }
    for j in 0..d {
        std[j] = (std[j] / (n_elite as f32)).sqrt() + 1e-6;
    }

    let dev = candidates.device();
    Ok((
        Tensor::from_vec(mean, d, dev)?,
        Tensor::from_vec(std, d, dev)?,
    ))
}

/// Metropolis-Hastings acceptance probability: min(1, exp(-delta_e / T)).
pub fn sa_acceptance_prob(delta_e: &Tensor, temp: f64) -> Result<Tensor> {
    let de_vec = delta_e.to_vec1::<f32>()?;
    let mut probs = vec![0.0f32; de_vec.len()];
    for i in 0..de_vec.len() {
        let val = (-(de_vec[i] as f64) / temp).exp();
        probs[i] = val.min(1.0) as f32;
    }
    Tensor::from_vec(probs, de_vec.len(), delta_e.device())
}

/// Replica exchange swap probability: min(1, exp((beta1 - beta2) * (e1 - e2))).
pub fn parallel_tempering_swap_prob(e1: f64, e2: f64, temp1: f64, temp2: f64) -> f64 {
    let beta1 = 1.0 / temp1;
    let beta2 = 1.0 / temp2;
    let delta = (beta1 - beta2) * (e1 - e2);
    delta.exp().min(1.0)
}

/// Unified module encapsulating combinatorial optimization routines for parity checks.
pub struct CombinatorialUnifiedModule {
    pub log_alpha: Tensor,
    pub candidates: Tensor,
    pub scores: Tensor,
    pub delta_e: Tensor,
    pub sa_temp: f64,
    pub pt_e1: f64,
    pub pt_e2: f64,
    pub pt_t1: f64,
    pub pt_t2: f64,
    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_sinkhorn_out: Tensor,
    pub golden_cem_mean: Tensor,
    pub golden_cem_std: Tensor,
    pub golden_sa_probs: Tensor,
    pub golden_pt_swap_prob: f64,
    pub golden_anchor_out: Tensor,
}

impl CombinatorialUnifiedModule {
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(path.as_ref(), &dev)?;

        let log_alpha = tensors
            .get("log_alpha")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing log_alpha".into()))?;
        let candidates = tensors
            .get("candidates")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing candidates".into()))?;
        let scores = tensors
            .get("scores")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing scores".into()))?;
        let delta_e = tensors
            .get("delta_e")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing delta_e".into()))?;

        let sa_temp = tensors
            .get("sa_temp")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing sa_temp".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let pt_e1 = tensors
            .get("pt_e1")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing pt_e1".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let pt_e2 = tensors
            .get("pt_e2")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing pt_e2".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let pt_t1 = tensors
            .get("pt_t1")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing pt_t1".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let pt_t2 = tensors
            .get("pt_t2")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing pt_t2".into()))?
            .to_vec1::<f32>()?[0] as f64;

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

        let golden_sinkhorn_out = tensors
            .get("golden_sinkhorn_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sinkhorn_out".into()))?;
        let golden_cem_mean = tensors
            .get("golden_cem_mean")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_cem_mean".into()))?;
        let golden_cem_std = tensors
            .get("golden_cem_std")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_cem_std".into()))?;
        let golden_sa_probs = tensors
            .get("golden_sa_probs")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_sa_probs".into()))?;
        let golden_pt_swap_prob = tensors
            .get("golden_pt_swap_prob")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_pt_swap_prob".into()))?
            .to_vec1::<f32>()?[0] as f64;
        let golden_anchor_out = tensors
            .get("golden_anchor_out")
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("Missing golden_anchor_out".into()))?;

        Ok(Self {
            log_alpha,
            candidates,
            scores,
            delta_e,
            sa_temp,
            pt_e1,
            pt_e2,
            pt_t1,
            pt_t2,
            anchor,
            anchor_x,
            golden_sinkhorn_out,
            golden_cem_mean,
            golden_cem_std,
            golden_sa_probs,
            golden_pt_swap_prob,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errors = BTreeMap::new();

        // 1. Sinkhorn Invariant
        let sinkhorn_out = run_sinkhorn(&self.log_alpha, 10, 0.5)?;
        let err_sinkhorn = max_abs_diff(&sinkhorn_out, &self.golden_sinkhorn_out)?;
        if err_sinkhorn >= atol {
            candle_core::bail!(
                "Sinkhorn invariant failed: err={:e} >= {:e}",
                err_sinkhorn,
                atol
            );
        }
        errors.insert("sinkhorn_err".to_string(), err_sinkhorn);

        // 2. CEM Invariant
        let (mean, std) = step_cem(&self.candidates, &self.scores, 0.25)?;
        let err_mean = max_abs_diff(&mean, &self.golden_cem_mean)?;
        let err_std = max_abs_diff(&std, &self.golden_cem_std)?;
        let err_cem = err_mean.max(err_std);
        if err_cem >= atol {
            candle_core::bail!("CEM invariant failed: err={:e} >= {:e}", err_cem, atol);
        }
        errors.insert("cem_err".to_string(), err_cem);

        // 3. Simulated Annealing Invariant
        let sa_probs = sa_acceptance_prob(&self.delta_e, self.sa_temp)?;
        let err_sa = max_abs_diff(&sa_probs, &self.golden_sa_probs)?;
        if err_sa >= atol {
            candle_core::bail!("SA invariant failed: err={:e} >= {:e}", err_sa, atol);
        }
        errors.insert("sa_err".to_string(), err_sa);

        // 4. Parallel Tempering Invariant
        let swap_prob =
            parallel_tempering_swap_prob(self.pt_e1, self.pt_e2, self.pt_t1, self.pt_t2);
        let err_pt = (swap_prob - self.golden_pt_swap_prob).abs();
        if err_pt >= atol {
            candle_core::bail!("PT invariant failed: err={:e} >= {:e}", err_pt, atol);
        }
        errors.insert("pt_err".to_string(), err_pt);

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
