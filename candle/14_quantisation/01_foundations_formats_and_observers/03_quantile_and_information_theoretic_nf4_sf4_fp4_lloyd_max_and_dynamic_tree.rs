//! Quantile and Information Theoretic NF4 SF4 FP4 Lloyd Max and Dynamic Tree
//! in candle/14_quantisation/01_foundations_formats_and_observers.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

pub const SF4_LEVELS: [f32; 16] = [
    -1.0, -0.7303, -0.5512, -0.4072, -0.2885, -0.1852, -0.0905, 0.0, 0.0792, 0.1614, 0.2486,
    0.3448, 0.4571, 0.5982, 0.7681, 1.0,
];

fn quantize_to_codebook(
    x: &Tensor,
    block_size: usize,
    levels: &[f32; 16],
) -> Result<(Tensor, Tensor)> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut dequant_vec = Vec::with_capacity(vec.len());
    let mut scales_vec = Vec::new();

    for chunk in vec.chunks(block_size) {
        let amax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        if amax == 0.0 {
            scales_vec.push(1.0f32);
            for _ in 0..chunk.len() {
                dequant_vec.push(0.0f32);
            }
        } else {
            scales_vec.push(amax);
            let inv_scale = 1.0f32 / amax;

            for &v in chunk {
                let norm_v = v * inv_scale;
                let mut best_level = levels[0];
                let mut min_diff = (norm_v - levels[0]).abs();

                for &lvl in &levels[1..] {
                    let diff = (norm_v - lvl).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        best_level = lvl;
                    }
                }
                dequant_vec.push(best_level * amax);
            }
        }
    }

    let n_scales = scales_vec.len();
    let dequant = Tensor::from_vec(dequant_vec, x.shape(), x.device())?;
    let scales = Tensor::from_vec(scales_vec, &[n_scales], x.device())?;
    Ok((dequant, scales))
}

/// NormalFloat 4 (NF4) Block Quantizer.
pub fn nf4_quantize_dequantize(x: &Tensor, block_size: usize) -> Result<(Tensor, Tensor)> {
    quantize_to_codebook(x, block_size, &NF4_LEVELS)
}

/// Student-t Float 4 (SF4) Block Quantizer.
pub fn sf4_quantize_dequantize(x: &Tensor, block_size: usize) -> Result<(Tensor, Tensor)> {
    quantize_to_codebook(x, block_size, &SF4_LEVELS)
}

/// Lloyd-Max Optimal Scalar Quantizer.
pub fn lloyd_max_scalar_quantize(
    data: &Tensor,
    num_levels: usize,
    iters: usize,
) -> Result<(Tensor, Tensor)> {
    let flat = data.flatten_all()?.to_vec1::<f32>()?;
    let dmin = flat.iter().fold(f32::INFINITY, |acc, &v| acc.min(v));
    let dmax = flat.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    let mut centroids = Vec::with_capacity(num_levels);
    for k in 0..num_levels {
        let frac = (k as f32) / ((num_levels - 1) as f32);
        centroids.push(dmin + frac * (dmax - dmin));
    }

    for _ in 0..iters {
        let mut sums = vec![0.0f32; num_levels];
        let mut counts = vec![0usize; num_levels];

        for &v in &flat {
            let mut best_k = 0;
            let mut min_diff = (v - centroids[0]).abs();
            for (k, &c) in centroids.iter().enumerate().skip(1) {
                let diff = (v - c).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_k = k;
                }
            }
            sums[best_k] += v;
            counts[best_k] += 1;
        }

        for k in 0..num_levels {
            if counts[k] > 0 {
                centroids[k] = sums[k] / (counts[k] as f32);
            }
        }
    }

    let mut q_vec = Vec::with_capacity(flat.len());
    for &v in &flat {
        let mut best_c = centroids[0];
        let mut min_diff = (v - centroids[0]).abs();
        for &c in centroids.iter().skip(1) {
            let diff = (v - c).abs();
            if diff < min_diff {
                min_diff = diff;
                best_c = c;
            }
        }
        q_vec.push(best_c);
    }

    let n_centroids = centroids.len();
    let q_out = Tensor::from_vec(q_vec, data.shape(), data.device())?;
    let c_out = Tensor::from_vec(centroids, &[n_centroids], data.device())?;
    Ok((q_out, c_out))
}

/// Dynamic Tree Scalar Quantizer.
pub fn dynamic_tree_scalar_quantize(x: &Tensor, depth: usize) -> Result<Tensor> {
    let mut flat = vec![x.flatten_all()?.to_vec1::<f32>()?];

    for _ in 0..depth {
        let mut next_flat = Vec::new();
        for leaf in flat {
            if leaf.is_empty() {
                next_flat.push(Vec::new());
                next_flat.push(Vec::new());
                continue;
            }
            let mut sorted = leaf.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[(sorted.len() - 1) / 2];

            let left: Vec<f32> = leaf.iter().cloned().filter(|&v| v <= median).collect();
            let right: Vec<f32> = leaf.iter().cloned().filter(|&v| v > median).collect();
            next_flat.push(left);
            next_flat.push(right);
        }
        flat = next_flat;
    }

    let centroids: Vec<f32> = flat
        .into_iter()
        .map(|l| {
            if l.is_empty() {
                0.0f32
            } else {
                l.iter().sum::<f32>() / (l.len() as f32)
            }
        })
        .collect();

    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(vec.len());
    for &v in &vec {
        let mut best_c = centroids[0];
        let mut min_diff = (v - centroids[0]).abs();
        for &c in centroids.iter().skip(1) {
            let diff = (v - c).abs();
            if diff < min_diff {
                min_diff = diff;
                best_c = c;
            }
        }
        out.push(best_c);
    }

    Tensor::from_vec(out, x.shape(), x.device())
}
