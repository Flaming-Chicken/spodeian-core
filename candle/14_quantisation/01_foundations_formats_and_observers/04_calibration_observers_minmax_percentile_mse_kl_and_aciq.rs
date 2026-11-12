//! Calibration Observers Minmax Percentile Mse KL and Aciq
//! in candle/14_quantisation/01_foundations_formats_and_observers.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Symmetric uniform quantize and dequantize with scale = alpha / qmax.
pub fn quantize_dequantize_symmetric(x: &Tensor, alpha: f32, qmax: f32) -> Result<Tensor> {
    if alpha == 0.0 {
        return Tensor::zeros(x.shape(), x.dtype(), x.device());
    }
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let scale = alpha / qmax;
    let inv_scale = 1.0f32 / scale;

    let out: Vec<f32> = flat
        .into_iter()
        .map(|v| (v * inv_scale).round().clamp(-qmax, qmax) * scale)
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}

/// MinMax Observer: computes exact min, max, amax, and symmetric dequantized tensor.
pub fn minmax_observer(x: &Tensor, qmax: f32) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let amax = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let dequant = quantize_dequantize_symmetric(x, amax, qmax)?;
    let stats = Tensor::from_vec(vec![min_val, max_val, amax], &[3], x.device())?;
    Ok((dequant, stats))
}

/// Moving Average MinMax Observer: exponential moving average across batch dimension.
pub fn moving_average_minmax_observer(batches: &Tensor, momentum: f32) -> Result<Tensor> {
    let dims = batches.dims();
    let num_batches = dims[0];
    let batch_size = dims[1];
    let flat = batches.flatten_all()?.to_vec1::<f32>()?;

    let first_chunk = &flat[0..batch_size];
    let mut running_min = first_chunk.iter().cloned().fold(f32::INFINITY, f32::min);
    let mut running_max = first_chunk
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    for b in 1..num_batches {
        let chunk = &flat[b * batch_size..(b + 1) * batch_size];
        let b_min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
        let b_max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        running_min = (1.0 - momentum) * running_min + momentum * b_min;
        running_max = (1.0 - momentum) * running_max + momentum * b_max;
    }

    Tensor::from_vec(vec![running_min, running_max], &[2], batches.device())
}

/// Percentile Observer: clips dynamic range to specified absolute value percentile.
pub fn percentile_observer(x: &Tensor, percentile: f32, qmax: f32) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let mut sorted_abs: Vec<f32> = flat.iter().map(|v| v.abs()).collect();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let k = (percentile * ((sorted_abs.len() - 1) as f32)) as usize;
    let alpha = sorted_abs[k];

    let dequant = quantize_dequantize_symmetric(x, alpha, qmax)?;
    let alpha_tensor = Tensor::from_vec(vec![alpha], &[1], x.device())?;
    Ok((dequant, alpha_tensor))
}

/// MSE Observer: candidate grid search minimizing reconstruction MSE.
pub fn mse_observer(x: &Tensor, num_candidates: usize, qmax: f32) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let amax = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let mut best_mse = f32::INFINITY;
    let mut best_alpha = amax;

    for k in 0..num_candidates {
        let candidate = amax * (0.55 + 0.05 * (k as f32));
        let scale = candidate / qmax;
        let inv_scale = 1.0f32 / scale;

        let mut mse = 0.0f32;
        for &v in &flat {
            let q = (v * inv_scale).round().clamp(-qmax, qmax) * scale;
            let diff = v - q;
            mse += diff * diff;
        }
        mse /= flat.len() as f32;

        if mse < best_mse {
            best_mse = mse;
            best_alpha = candidate;
        }
    }

    let dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)?;
    let alpha_tensor = Tensor::from_vec(vec![best_alpha], &[1], x.device())?;
    Ok((dequant, alpha_tensor))
}

/// KL Divergence Observer: TensorRT-style discrete entropy calibration.
pub fn kl_divergence_observer(
    x: &Tensor,
    num_bins: usize,
    num_quant_bins: usize,
    qmax: f32,
) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let amax = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let bin_width = amax / (num_bins as f32);

    let mut hist = vec![0.0f32; num_bins];
    for &v in &flat {
        let b = ((v.abs() / bin_width) as usize).min(num_bins - 1);
        hist[b] += 1.0;
    }

    let mut best_kl = f32::INFINITY;
    let mut best_t = num_bins;

    let mut t = num_quant_bins;
    while t <= num_bins {
        let mut p = hist[0..t].to_vec();
        let tail_sum: f32 = hist[t..].iter().sum();
        p[t - 1] += tail_sum;

        let bin_size = t / num_quant_bins;
        let mut q = vec![0.0f32; t];

        for k in 0..num_quant_bins {
            let start = k * bin_size;
            let end = (k + 1) * bin_size;
            let chunk = &p[start..end];
            let nonzero = chunk.iter().filter(|&&v| v > 0.0).count();
            if nonzero > 0 {
                let mean_val = chunk.iter().sum::<f32>() / (nonzero as f32);
                for i in start..end {
                    if p[i] > 0.0 {
                        q[i] = mean_val;
                    }
                }
            }
        }

        let p_sum: f32 = p.iter().map(|&v| v + 1e-7).sum();
        let q_sum: f32 = q.iter().map(|&v| v + 1e-7).sum();

        let mut kl = 0.0f32;
        for i in 0..t {
            let p_norm = (p[i] + 1e-7) / p_sum;
            let q_norm = (q[i] + 1e-7) / q_sum;
            kl += p_norm * (p_norm / q_norm).ln();
        }

        if kl < best_kl {
            best_kl = kl;
            best_t = t;
        }

        t += 8;
    }

    let best_alpha = (best_t as f32 / num_bins as f32) * amax;
    let dequant = quantize_dequantize_symmetric(x, best_alpha, qmax)?;
    let alpha_tensor = Tensor::from_vec(vec![best_alpha], &[1], x.device())?;
    Ok((dequant, alpha_tensor))
}

/// ACIQ Gaussian Observer: optimal analytical threshold for Gaussian distribution.
pub fn aciq_gaussian_observer(x: &Tensor, qmax: f32) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let n = flat.len() as f32;
    let mu = flat.iter().sum::<f32>() / n;
    let variance = flat.iter().map(|&v| (v - mu) * (v - mu)).sum::<f32>() / n;
    let sigma = variance.sqrt();
    let beta = 2.8279f32;
    let alpha = beta * sigma;

    let dequant = quantize_dequantize_symmetric(x, alpha, qmax)?;
    let alpha_tensor = Tensor::from_vec(vec![alpha], &[1], x.device())?;
    Ok((dequant, alpha_tensor))
}

/// ACIQ Laplace Observer: optimal analytical threshold for Laplace distribution.
pub fn aciq_laplace_observer(x: &Tensor, qmax: f32) -> Result<(Tensor, Tensor)> {
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let n = flat.len() as f32;
    let mu = flat.iter().sum::<f32>() / n;
    let b = flat.iter().map(|&v| (v - mu).abs()).sum::<f32>() / n;
    let beta = 3.89f32;
    let alpha = beta * b;

    let dequant = quantize_dequantize_symmetric(x, alpha, qmax)?;
    let alpha_tensor = Tensor::from_vec(vec![alpha], &[1], x.device())?;
    Ok((dequant, alpha_tensor))
}
