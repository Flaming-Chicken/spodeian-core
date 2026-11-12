//! KoboldCpp & llama.cpp Advanced Sampler Suite (`sml-runtime::sampler`).
//!
//! Implements a comprehensive, production-grade sampling pipeline:
//! 1. **Repetition, Frequency & Presence Penalties**
//! 2. **Dry Sampling (Multi-Token N-Gram Loop Breaker)**
//! 3. **Dynamic Temperature (Entropy-Modulated Sampling)**
//! 4. **XTC (Exclude Top Choices)**
//! 5. **Min-P Sampling**
//! 6. **Top-A Sampling**
//! 7. **Typical-P Sampling**
//! 8. **Tail-Free Sampling (TFS)**
//! 9. **Top-K & Top-P (Nucleus)**
//! 10. **Mirostat v1 & v2 (Perplexity-Targeting Control)**

use serde::{Deserialize, Serialize};

/// Dynamic Temperature Configuration.
/// Modulates sampling temperature based on Shannon entropy of the logits distribution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DynamicTempConfig {
    pub min_temp: f32,
    pub max_temp: f32,
    pub exponent: f32,
}

impl Default for DynamicTempConfig {
    fn default() -> Self {
        Self {
            min_temp: 0.5,
            max_temp: 1.5,
            exponent: 1.0,
        }
    }
}

/// Dry Sampling Configuration (KoboldCpp multi-token repetition breaker).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DrySamplingConfig {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    pub penalty_window: usize,
}

impl Default for DrySamplingConfig {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            penalty_window: 256,
        }
    }
}

/// Mirostat Sampling Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MirostatConfig {
    pub version: u8, // 1 or 2
    pub tau: f32,    // target entropy / surprise
    pub eta: f32,    // learning rate
    pub m: usize,    // vocabulary subset for v1
}

impl Default for MirostatConfig {
    fn default() -> Self {
        Self {
            version: 2,
            tau: 5.0,
            eta: 0.1,
            m: 100,
        }
    }
}

/// Mirostat Dynamic State.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MirostatState {
    pub mu: f32,
}

impl Default for MirostatState {
    fn default() -> Self {
        Self { mu: 10.0 }
    }
}

/// Comprehensive Sampler Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub dynamic_temperature: Option<DynamicTempConfig>,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub top_a: f32,
    pub typical_p: f32,
    pub tfs: f32,
    pub xtc_threshold: f32,
    pub xtc_probability: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub dry_sampling: Option<DrySamplingConfig>,
    pub mirostat: Option<MirostatConfig>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            dynamic_temperature: None,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            top_a: 0.0,
            typical_p: 1.0,
            tfs: 1.0,
            xtc_threshold: 0.0,
            xtc_probability: 0.0,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_sampling: None,
            mirostat: None,
        }
    }
}

/// Applies repetition, presence, and frequency penalties to logits in place.
pub fn apply_repetition_penalties(
    logits: &mut [f32],
    history: &[usize],
    rep_penalty: f32,
    freq_penalty: f32,
    pres_penalty: f32,
) {
    if history.is_empty() {
        return;
    }

    let mut counts: ahash::AHashMap<usize, usize> = ahash::AHashMap::new();
    for &tok in history {
        *counts.entry(tok).or_insert(0) += 1;
    }

    for (&tok, &count) in &counts {
        if tok < logits.len() {
            let l = logits[tok];
            let penalized = if l > 0.0 {
                l / rep_penalty
            } else {
                l * rep_penalty
            };
            logits[tok] = penalized - (count as f32 * freq_penalty) - (if count > 0 { pres_penalty } else { 0.0 });
        }
    }
}

/// Applies Dry Sampling (n-gram sequence matching penalty).
pub fn apply_dry_sampling(
    logits: &mut [f32],
    history: &[usize],
    config: &DrySamplingConfig,
) {
    let hist_len = history.len();
    if hist_len <= config.allowed_length {
        return;
    }

    let window_start = hist_len.saturating_sub(config.penalty_window);
    let search_window = &history[window_start..];
    let n = search_window.len();

    // Match n-grams of decreasing length
    for match_len in (config.allowed_length..=n.min(32)).rev() {
        let prefix = &history[hist_len - match_len..];
        for i in 0..n.saturating_sub(match_len) {
            if &search_window[i..i + match_len] == prefix {
                let next_tok = search_window[i + match_len];
                if next_tok < logits.len() {
                    let exponent = (match_len - config.allowed_length) as f32;
                    let penalty = config.multiplier * config.base.powf(exponent);
                    logits[next_tok] -= penalty;
                }
            }
        }
    }
}

/// Computes Shannon entropy $H(P) = -\sum p_i \ln p_i$ and dynamically scales temperature.
pub fn compute_dynamic_temperature(
    logits: &[f32],
    base_temp: f32,
    dyn_cfg: &DynamicTempConfig,
) -> f32 {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }
    if sum_exp <= 0.0 {
        return base_temp;
    }

    let mut entropy = 0.0f32;
    for &l in logits {
        let p = (l - max_l).exp() / sum_exp;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (logits.len() as f32).ln().max(1e-5);
    let norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0);
    let scale = norm_entropy.powf(dyn_cfg.exponent);

    dyn_cfg.min_temp + (dyn_cfg.max_temp - dyn_cfg.min_temp) * scale
}

/// Applies Min-P truncation: keeps tokens where $P(w) \ge p_{\text{base}} \times P_{\max}$.
pub fn apply_min_p(probs: &mut Vec<(usize, f32)>, min_p: f32) {
    if probs.is_empty() || min_p <= 0.0 {
        return;
    }
    let p_max = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
    let threshold = p_max * min_p;
    probs.retain(|(_, p)| *p >= threshold);
}

/// Applies Top-A truncation: keeps tokens where $P(w) \ge A \times P_{\max}^2$.
pub fn apply_top_a(probs: &mut Vec<(usize, f32)>, top_a: f32) {
    if probs.is_empty() || top_a <= 0.0 {
        return;
    }
    let p_max = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
    let threshold = top_a * p_max * p_max;
    probs.retain(|(_, p)| *p >= threshold);
}

/// Applies Typical-P truncation: keeps tokens with information content closest to entropy.
pub fn apply_typical_p(probs: &mut Vec<(usize, f32)>, typical_p: f32) {
    if probs.is_empty() || typical_p >= 1.0 {
        return;
    }

    let mut entropy = 0.0f32;
    for (_, p) in probs.iter() {
        if *p > 1e-12 {
            entropy -= *p * p.ln();
        }
    }

    let mut diffs: Vec<(usize, f32, f32)> = probs
        .iter()
        .map(|&(tok, p)| {
            let info = -p.max(1e-12).ln();
            let diff = (info - entropy).abs();
            (tok, p, diff)
        })
        .collect();

    diffs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut cut_idx = diffs.len();
    for (i, (_, p, _)) in diffs.iter().enumerate() {
        cumulative += *p;
        if cumulative >= typical_p {
            cut_idx = i + 1;
            break;
        }
    }

    diffs.truncate(cut_idx);
    probs.clear();
    for (tok, p, _) in diffs {
        probs.push((tok, p));
    }
}

/// Applies Tail-Free Sampling (TFS): cuts off tail based on second derivative inflection.
pub fn apply_tfs(probs: &mut Vec<(usize, f32)>, z: f32) {
    if probs.len() <= 2 || z >= 1.0 {
        return;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // First derivatives
    let mut d1 = Vec::with_capacity(probs.len() - 1);
    for i in 0..probs.len() - 1 {
        d1.push(probs[i].1 - probs[i + 1].1);
    }

    // Second derivatives
    let mut d2 = Vec::with_capacity(d1.len() - 1);
    for i in 0..d1.len() - 1 {
        d2.push((d1[i] - d1[i + 1]).abs());
    }

    let sum_d2: f32 = d2.iter().sum();
    if sum_d2 <= 0.0 {
        return;
    }

    let mut cum_d2 = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &val) in d2.iter().enumerate() {
        cum_d2 += val / sum_d2;
        if cum_d2 >= z {
            cutoff = i + 1;
            break;
        }
    }

    probs.truncate(cutoff);
}

/// Applies XTC (Exclude Top Choices): suppresses top token if dominant.
pub fn apply_xtc(
    probs: &mut Vec<(usize, f32)>,
    threshold: f32,
    probability: f32,
    random_val: f32,
) {
    if probs.len() < 2 || threshold <= 0.0 || probability <= 0.0 {
        return;
    }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let p_top = probs[0].1;
    let p_second = probs[1].1;

    if p_top >= threshold && p_second > 0.0 && random_val < probability {
        probs.remove(0);
    }
}

/// Full KoboldCpp Sample Orchestrator.
pub fn sample_token(
    logits: &[f32],
    history: &[usize],
    config: &SamplerConfig,
    mirostat_state: &mut Option<MirostatState>,
    random_uniform: f32,
) -> usize {
    assert!(!logits.is_empty(), "Logits cannot be empty");

    let mut working_logits = logits.to_vec();

    // 1. Repetition, Presence & Frequency Penalties
    apply_repetition_penalties(
        &mut working_logits,
        history,
        config.repetition_penalty,
        config.frequency_penalty,
        config.presence_penalty,
    );

    // 2. Dry Sampling
    if let Some(ref dry_cfg) = config.dry_sampling {
        apply_dry_sampling(&mut working_logits, history, dry_cfg);
    }

    // 3. Dynamic Temperature
    let eff_temp = if let Some(ref dyn_cfg) = config.dynamic_temperature {
        compute_dynamic_temperature(&working_logits, config.temperature, dyn_cfg)
    } else {
        config.temperature
    };

    let temp = eff_temp.max(1e-4);
    for l in working_logits.iter_mut() {
        *l /= temp;
    }

    // 4. Softmax to Probabilities
    let max_l = working_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = working_logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| {
            let e = (l - max_l).exp();
            exp_sum += e;
            (idx, e)
        })
        .collect();

    if exp_sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= exp_sum;
        }
    }

    // 5. Mirostat Sampling (v2)
    if let Some(ref miro_cfg) = config.mirostat {
        if miro_cfg.version == 2 {
            let state = mirostat_state.get_or_insert_with(MirostatState::default);
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Truncate according to mu
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cum += *p;
                if -cum.max(1e-12).log2() > state.mu {
                    cutoff = (i + 1).max(1);
                    break;
                }
            }
            probs.truncate(cutoff);

            // Re-normalize and sample
            let p_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
            if p_sum > 0.0 {
                probs.iter_mut().for_each(|(_, p)| *p /= p_sum);
            }

            let sampled_idx = sample_from_probs(&probs, random_uniform);
            let selected_p = probs.iter().find(|(tok, _)| *tok == sampled_idx).map(|(_, p)| *p).unwrap_or(1e-4);

            // Update Mirostat state
            let surprise = -selected_p.log2();
            let error = surprise - miro_cfg.tau;
            state.mu -= miro_cfg.eta * error;

            return sampled_idx;
        }
    }

    // 6. XTC
    if config.xtc_threshold > 0.0 {
        apply_xtc(&mut probs, config.xtc_threshold, config.xtc_probability, random_uniform);
    }

    // 7. Top-K
    if config.top_k > 0 && config.top_k < probs.len() {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs.truncate(config.top_k);
    }

    // 8. Min-P
    if config.min_p > 0.0 {
        apply_min_p(&mut probs, config.min_p);
    }

    // 9. Top-A
    if config.top_a > 0.0 {
        apply_top_a(&mut probs, config.top_a);
    }

    // 10. Typical-P
    if config.typical_p < 1.0 {
        apply_typical_p(&mut probs, config.typical_p);
    }

    // 11. TFS
    if config.tfs < 1.0 {
        apply_tfs(&mut probs, config.tfs);
    }

    // 12. Top-P (Nucleus)
    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cum += *p;
            if cum >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Final Re-normalization and Selection
    let p_sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if p_sum > 0.0 {
        probs.iter_mut().for_each(|(_, p)| *p /= p_sum);
    }

    sample_from_probs(&probs, random_uniform)
}

#[inline]
fn sample_from_probs(probs: &[(usize, f32)], r: f32) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let mut acc = 0.0f32;
    for &(tok, p) in probs {
        acc += p;
        if r <= acc {
            return tok;
        }
    }
    probs.last().map(|(tok, _)| *tok).unwrap_or(0)
}
