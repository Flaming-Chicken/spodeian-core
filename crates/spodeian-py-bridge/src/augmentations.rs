/// Mathematical primitives and data augmentations shared between Rust and PyTorch DataLoaders.

/// Random sequence masking for Self-Supervised Learning (SSL).
/// Replaces random segments of length proportional to `mask_ratio` with `mask_value`.
pub fn apply_temporal_mask(sequence: &[f64], mask_ratio: f64, mask_value: f64, seed: u64) -> Vec<f64> {
    if sequence.is_empty() {
        return Vec::new();
    }
    let ratio = mask_ratio.clamp(0.0, 1.0);
    let mut result = sequence.to_vec();
    let mut rng_state = if seed == 0 { 0x853c49e6748fea9b } else { seed };

    for val in result.iter_mut() {
        // Linear congruential pseudorandom generator for deterministic reproducible masking
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

        // Box-Muller transform
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
