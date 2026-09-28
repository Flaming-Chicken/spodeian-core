//! Pure Rust quantitative financial factors, time-series transforms, and data augmentations.
//!
//! Centralized in `spodeian-ml-utils` to guarantee bit-perfect mathematical parity
//! between Python/PyTorch data augmentations (`spodeian-py-bridge`) and Rust production trading engines (`Risky Business`).

/// Computes log returns: ln(P_t / P_{t-1}). First element is 0.0.
pub fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![0.0; prices.len()];
    }
    let mut returns = Vec::with_capacity(prices.len());
    returns.push(0.0);

    for i in 1..prices.len() {
        let p_prev = prices[i - 1];
        let p_curr = prices[i];
        if p_prev > 0.0 && p_curr > 0.0 {
            returns.push((p_curr / p_prev).ln());
        } else {
            returns.push(0.0);
        }
    }

    returns
}

/// Computes rolling window sample volatility annualized with sqrt(252).
pub fn calculate_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.is_empty() || window == 0 {
        return Vec::new();
    }
    let annualization = 252.0_f64.sqrt();
    let mut vol = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &returns[start..=i];
        let n = slice.len() as f64;
        if n < 2.0 {
            vol.push(0.0);
            continue;
        }
        let mean = slice.iter().sum::<f64>() / n;
        let var = slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        vol.push(var.sqrt() * annualization);
    }

    vol
}

/// Computes Exponential Moving Average (EMA) with span alpha = 2 / (span + 1).
pub fn calculate_ema(series: &[f64], span: usize) -> Vec<f64> {
    if series.is_empty() {
        return Vec::new();
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ema = Vec::with_capacity(series.len());
    let mut prev = series[0];
    ema.push(prev);

    for &val in &series[1..] {
        prev = alpha * val + (1.0 - alpha) * prev;
        ema.push(prev);
    }

    ema
}

/// Computes cumulative Volume-Weighted Average Price (VWAP).
pub fn calculate_vwap(prices: &[f64], volumes: &[f64]) -> Vec<f64> {
    let n = prices.len().min(volumes.len());
    if n == 0 {
        return Vec::new();
    }

    let mut vwap = Vec::with_capacity(n);
    let mut cum_pv = 0.0;
    let mut cum_v = 0.0;

    for i in 0..n {
        cum_pv += prices[i] * volumes[i];
        cum_v += volumes[i];
        if cum_v > 0.0 {
            vwap.push(cum_pv / cum_v);
        } else {
            vwap.push(prices[i]);
        }
    }

    vwap
}

/// Computes rolling window Z-score normalisation: (x - mean) / (std + 1e-8).
pub fn rolling_zscore_normalize(sequence: &[f64], window: usize) -> Vec<f64> {
    if sequence.is_empty() || window == 0 {
        return Vec::new();
    }
    let mut normalized = Vec::with_capacity(sequence.len());

    for i in 0..sequence.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &sequence[start..=i];
        let n = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / n;
        let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = (variance + 1e-8).sqrt();
        normalized.push((sequence[i] - mean) / std);
    }

    normalized
}

/// Random sequence masking for Self-Supervised Learning (SSL).
pub fn apply_temporal_mask(sequence: &[f64], mask_ratio: f64, mask_value: f64, seed: u64) -> Vec<f64> {
    if sequence.is_empty() {
        return Vec::new();
    }
    let ratio = mask_ratio.clamp(0.0, 1.0);
    let mut result = sequence.to_vec();
    let mut rng_state = if seed == 0 { 0x853c49e6748fea9b } else { seed };

    for val in result.iter_mut() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rand_float = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64);
        if rand_float < ratio {
            *val = mask_value;
        }
    }

    result
}

/// Injects Gaussian/pseudorandom jitter noise scaled by `std_dev` using Box-Muller transform.
pub fn apply_gaussian_jitter(sequence: &[f64], std_dev: f64, seed: u64) -> Vec<f64> {
    if sequence.is_empty() || std_dev <= 0.0 {
        return sequence.to_vec();
    }
    let mut result = Vec::with_capacity(sequence.len());
    let mut rng_state = if seed == 0 { 0xda942042e4dd58b5 } else { seed };

    let mut i = 0;
    while i < sequence.len() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (((rng_state >> 33) as f64) + 1.0) / (((1u64 << 31) as f64) + 2.0);
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (((rng_state >> 33) as f64) + 1.0) / (((1u64 << 31) as f64) + 2.0);

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        result.push(sequence[i] + z0 * std_dev);
        i += 1;

        if i < sequence.len() {
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            result.push(sequence[i] + z1 * std_dev);
            i += 1;
        }
    }

    result
}

/// Linear interpolation time-warping to resample sequence to `target_len`.
pub fn time_warp_resample(sequence: &[f64], target_len: usize) -> Vec<f64> {
    if sequence.is_empty() || target_len == 0 {
        return Vec::new();
    }
    if sequence.len() == 1 || target_len == 1 {
        return vec![sequence[0]; target_len];
    }

    let mut resampled = Vec::with_capacity(target_len);
    let original_len = sequence.len();

    for i in 0..target_len {
        let t = (i as f64) * (original_len - 1) as f64 / (target_len - 1) as f64;
        let low = t.floor() as usize;
        let high = (low + 1).min(original_len - 1);
        let frac = t - (low as f64);
        let val = sequence[low] * (1.0 - frac) + sequence[high] * frac;
        resampled.push(val);
    }

    resampled
}
