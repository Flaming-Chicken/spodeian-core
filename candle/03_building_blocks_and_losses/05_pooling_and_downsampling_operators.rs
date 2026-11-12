//! Pooling and Downsampling Operators: GeM, BlurPool, PixelUnshuffle, and PMA (C03 Topic 05).
//!
//! Native Rust Candle implementation of:
//! 1. Generalized Mean Pooling (GeM) (Radenovic et al. 2018).
//! 2. Anti-Aliased Downsampling (BlurPool) (Zhang 2019).
//! 3. Lossless Spatial-to-Channel Downsampler (PixelUnshuffle) (Shi et al. 2016).
//! 4. Pooling by Multihead Attention (PMA) (Lee et al. 2019).
//! 5. Linear Anchor.

use candle_core::{Result, Tensor};

/// Linear anchor forward: Y = X @ W + b.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// Generalized Mean Pooling (GeM) forward: (1/|R| sum x^p)^(1/p) over spatial dimensions (H, W).
pub fn gem_pool_forward(x: &Tensor, p: f64, eps: f64) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
    let spatial_count = (h * w) as f32;

    let mut out = vec![vec![0.0f32; c]; b];
    for bi in 0..b {
        for ci in 0..c {
            let base = (bi * c + ci) * (h * w);
            let mut sum_p = 0.0f32;
            for i in 0..(h * w) {
                let val = x_flat[base + i].max(eps as f32);
                sum_p += val.powf(p as f32);
            }
            let mean_p = sum_p / spatial_count;
            out[bi][ci] = mean_p.powf((1.0 / p) as f32);
        }
    }
    Tensor::new(out, x.device())
}

/// Anti-Aliased Downsampling (BlurPool) with reflect padding and stride 2.
pub fn blur_pool_forward(x: &Tensor, kernel: &Tensor, stride: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
    let k_flat = kernel.flatten_all()?.to_vec1::<f32>()?; // [c * 1 * 3 * 3]

    let pad_h = h + 2;
    let pad_w = w + 2;
    let mut padded = vec![0.0f32; b * c * pad_h * pad_w];

    for bi in 0..b {
        for ci in 0..c {
            let in_base = (bi * c + ci) * (h * w);
            let pad_base = (bi * c + ci) * (pad_h * pad_w);

            for hi in 0..h {
                for wi in 0..w {
                    padded[pad_base + (hi + 1) * pad_w + (wi + 1)] = x_flat[in_base + hi * w + wi];
                }
            }
            for wi in 0..w {
                padded[pad_base + 0 * pad_w + (wi + 1)] = x_flat[in_base + 1 * w + wi];
                padded[pad_base + (pad_h - 1) * pad_w + (wi + 1)] =
                    x_flat[in_base + (h - 2) * w + wi];
            }
            for hi in 0..h {
                padded[pad_base + (hi + 1) * pad_w + 0] = x_flat[in_base + hi * w + 1];
                padded[pad_base + (hi + 1) * pad_w + (pad_w - 1)] =
                    x_flat[in_base + hi * w + (w - 2)];
            }
            padded[pad_base + 0 * pad_w + 0] = x_flat[in_base + 1 * w + 1];
            padded[pad_base + 0 * pad_w + (pad_w - 1)] = x_flat[in_base + 1 * w + (w - 2)];
            padded[pad_base + (pad_h - 1) * pad_w + 0] = x_flat[in_base + (h - 2) * w + 1];
            padded[pad_base + (pad_h - 1) * pad_w + (pad_w - 1)] =
                x_flat[in_base + (h - 2) * w + (w - 2)];
        }
    }

    let out_h = (pad_h - 3) / stride + 1;
    let out_w = (pad_w - 3) / stride + 1;
    let mut out = vec![0.0f32; b * c * out_h * out_w];

    for bi in 0..b {
        for ci in 0..c {
            let pad_base = (bi * c + ci) * (pad_h * pad_w);
            let out_base = (bi * c + ci) * (out_h * out_w);
            let k_base = ci * 9;

            for oh in 0..out_h {
                let sh = oh * stride;
                for ow in 0..out_w {
                    let sw = ow * stride;
                    let mut sum_val = 0.0f32;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let p_val = padded[pad_base + (sh + kh) * pad_w + (sw + kw)];
                            let k_val = k_flat[k_base + kh * 3 + kw];
                            sum_val += p_val * k_val;
                        }
                    }
                    out[out_base + oh * out_w + ow] = sum_val;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_h, out_w), x.device())
}

/// PixelUnshuffle: Rearranges elements from spatial dimensions into depth/channel dimension.
pub fn pixel_unshuffle_forward(x: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let r = downscale_factor;
    let out_h = h / r;
    let out_w = w / r;
    x.reshape((b, c, out_h, r, out_w, r))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * r * r, out_h, out_w))
}

/// Pooling by Multihead Attention (PMA): Learned seed query attending to input tokens.
pub fn pma_forward(x: &Tensor, s_seed: &Tensor) -> Result<Tensor> {
    let (b, _seq_len, dim) = x.dims3()?;
    let q = s_seed.broadcast_as((b, 1, dim))?;
    let scores = (q.matmul(&x.transpose(1, 2)?)? / (dim as f64).sqrt())?;
    let weights = candle_nn::ops::softmax(&scores, 2)?;
    let out = weights.matmul(x)?;
    out.squeeze(1)
}
