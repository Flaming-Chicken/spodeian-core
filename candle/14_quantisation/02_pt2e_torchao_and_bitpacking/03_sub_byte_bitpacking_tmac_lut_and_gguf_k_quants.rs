//! Sub Byte Bitpacking T-MAC LUT and GGUF K Quants
//! in candle/14_quantisation/02_pt2e_torchao_and_bitpacking.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Packs pairs of 4-bit unsigned integers [0, 15] into a single byte.
pub fn bitpack_2x4bit(v_int4: &Tensor) -> Result<Tensor> {
    let flat = v_int4.flatten_all()?.to_vec1::<f32>()?;
    let mut packed = Vec::with_capacity(flat.len() / 2);
    for chunk in flat.chunks(2) {
        let b0 = chunk[0] as u8 & 0x0F;
        let b1 = chunk[1] as u8 & 0x0F;
        packed.push(((b0) | (b1 << 4)) as f32);
    }
    Tensor::from_vec(packed, &[flat.len() / 2], v_int4.device())
}

/// Unpacks a byte array into interleaved 4-bit unsigned integers.
pub fn bitunpack_2x4bit(packed: &Tensor) -> Result<Tensor> {
    let flat = packed.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(flat.len() * 2);
    for &p in &flat {
        let b = p as u8;
        out.push((b & 0x0F) as f32);
        out.push(((b >> 4) & 0x0F) as f32);
    }
    Tensor::from_vec(out, &[flat.len() * 2], packed.device())
}

/// Packs groups of 5 ternary trits {0, 1, 2} into a single byte (3^5 = 243 < 256).
pub fn bitpack_5x_ternary(trits: &Tensor) -> Result<Tensor> {
    let flat = trits.flatten_all()?.to_vec1::<f32>()?;
    let mut packed = Vec::with_capacity(flat.len() / 5);
    for chunk in flat.chunks(5) {
        let code = (chunk[0] as i64)
            + 3 * (chunk[1] as i64)
            + 9 * (chunk[2] as i64)
            + 27 * (chunk[3] as i64)
            + 81 * (chunk[4] as i64);
        packed.push(code as f32);
    }
    Tensor::from_vec(packed, &[flat.len() / 5], trits.device())
}

/// Unpacks a base-3 packed byte array into groups of 5 ternary trits.
pub fn bitunpack_5x_ternary(packed_codes: &Tensor) -> Result<Tensor> {
    let flat = packed_codes.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(flat.len() * 5);
    for &code_f in &flat {
        let mut c = code_f as i64;
        for _ in 0..5 {
            out.push((c % 3) as f32);
            c /= 3;
        }
    }
    Tensor::from_vec(out, &[flat.len() * 5], packed_codes.device())
}

/// Precomputes 16-entry addition table for a group of 4 activations.
pub fn tmac_precompute_lut(activations_group4: &Tensor) -> Result<Tensor> {
    let a = activations_group4.flatten_all()?.to_vec1::<f32>()?;
    let mut lut = Vec::with_capacity(16);
    for i in 0..16 {
        let mut val = 0.0f32;
        for k in 0..4 {
            if (i >> k) & 1 == 1 {
                val += a[k];
            }
        }
        lut.push(val);
    }
    Tensor::from_vec(lut, &[16], activations_group4.device())
}

/// Computes multiplier-free matrix multiplication via direct table lookup.
pub fn tmac_lut_lookup(lut: &Tensor, weight_indices: &Tensor) -> Result<Tensor> {
    let l = lut.flatten_all()?.to_vec1::<f32>()?;
    let indices = weight_indices.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(indices.len());
    for &idx in &indices {
        out.push(l[idx as usize]);
    }
    Tensor::from_vec(out, weight_indices.shape(), weight_indices.device())
}

/// GGUF Q4_K_M hierarchical super-block quantization (256 elements in 8 sub-blocks).
pub fn gguf_q4_k_quantize_block(w_block256: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let flat = w_block256.flatten_all()?.to_vec1::<f32>()?;
    let amax = flat.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let mut d = amax / 127.0f32;
    if d < 1e-8 {
        d = 1e-8;
    }

    let mut sub_scales = Vec::with_capacity(8);
    let mut dequants = Vec::with_capacity(256);

    for chunk in flat.chunks(32) {
        let cmax = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let dk = if cmax > 0.0 {
            cmax / (7.0f32 * d)
        } else {
            1.0f32
        };
        let eff_scale = d * dk;
        let inv_scale = 1.0f32 / eff_scale;

        sub_scales.push(dk);
        for &v in chunk {
            let q = (v * inv_scale).round().clamp(-8.0, 7.0);
            dequants.push(q * eff_scale);
        }
    }

    let dequant_tensor = Tensor::from_vec(dequants, w_block256.shape(), w_block256.device())?;
    let super_scale_tensor = Tensor::from_vec(vec![d], &[1], w_block256.device())?;
    let sub_scales_tensor = Tensor::from_vec(sub_scales, &[8], w_block256.device())?;

    Ok((dequant_tensor, super_scale_tensor, sub_scales_tensor))
}
