//! Binary Sub1bit and Ternary BNN XNOR BiLLM OneBit TWN TTQ
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Binary Neural Network (BNN / XNOR-Net) 1-bit quantization.
pub fn bnn_xnor_quantize(w: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let ge = w.ge(0.0)?.to_dtype(DType::F32)?;
    let signs = ((&ge * 2.0)? - 1.0)?;
    let alpha = w.abs()?.mean_keepdim(0)?;
    let w_bnn = signs.broadcast_mul(&alpha)?;
    Ok((w_bnn, signs, alpha))
}

/// Ternary Weight Networks (TWN) 1.58-bit {-1, 0, +1} quantization.
pub fn twn_quantize(w: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let mean_abs = w.abs()?.mean_keepdim(0)?;
    let delta = (&mean_abs * 0.7)?;
    let delta_bc = delta.broadcast_as(w.shape())?;
    let gt = w.gt(&delta_bc)?.to_dtype(DType::F32)?;
    let neg_delta_bc = (&delta_bc * -1.0)?;
    let lt = w.lt(&neg_delta_bc)?.to_dtype(DType::F32)?;
    let t = (&gt - &lt)?;

    let nonzeros = t.abs()?;
    let nz_counts = nonzeros.sum_keepdim(0)?.clamp(1.0, f64::MAX)?;
    let alpha = (w.abs()?.mul(&nonzeros)?.sum_keepdim(0)? / nz_counts)?;
    let w_twn = t.broadcast_mul(&alpha)?;
    Ok((w_twn, t, alpha))
}

/// Trained Ternary Quantization (TTQ) with asymmetric positive and negative scales.
pub fn ttq_quantize(w: &Tensor, delta_factor: f64) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let max_abs = w.abs()?.max_keepdim(0)?;
    let delta = (&max_abs * delta_factor)?;
    let delta_bc = delta.broadcast_as(w.shape())?;

    let pos_mask = w.gt(&delta_bc)?.to_dtype(DType::F32)?;
    let neg_delta_bc = (&delta_bc * -1.0)?;
    let neg_mask = w.lt(&neg_delta_bc)?.to_dtype(DType::F32)?;
    let t = (&pos_mask - &neg_mask)?;

    let pos_count = pos_mask.sum_keepdim(0)?.clamp(1.0, f64::MAX)?;
    let neg_count = neg_mask.sum_keepdim(0)?.clamp(1.0, f64::MAX)?;

    let w_pos = (w.mul(&pos_mask)?.sum_keepdim(0)? / pos_count)?;
    let w_neg = (w.abs()?.mul(&neg_mask)?.sum_keepdim(0)? / neg_count)?;

    let w_ttq = (pos_mask.broadcast_mul(&w_pos)? - neg_mask.broadcast_mul(&w_neg)?)?;
    Ok((w_ttq, t, w_pos, w_neg))
}

/// OneBit: 1-bit value-sign rank-1 scaling decomposition: W = S * (g @ h)
pub fn onebit_quantize(w: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let ge = w.ge(0.0)?.to_dtype(DType::F32)?;
    let signs = ((&ge * 2.0)? - 1.0)?;
    let abs_w = w.abs()?;

    let g = abs_w.mean_keepdim(1)?;
    let total_abs = abs_w.sum_all()?.to_scalar::<f32>()?.max(1e-5);
    let (d_in, _) = w.dims2()?;
    let factor = (d_in as f64) / (total_abs as f64);
    let h = (abs_w.sum_keepdim(0)? * factor)?;

    let scale_matrix = g.matmul(&h)?;
    let w_onebit = signs.mul(&scale_matrix)?;
    Ok((w_onebit, signs, g, h))
}
