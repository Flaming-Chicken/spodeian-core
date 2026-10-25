//! Universal Quantization Kernels, SIMD Vectorization & Precision Matrix
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides optimized matrix multiplication, dequantization, and precision benchmarks
//! across all 12 formats: FP32, FP16, BF16, FP8 (E4M3 / E5M2), NF4, INT8, INT4,
//! Posit16, Posit8, Ternary 1.58b, Log8, and Binary (BNN).

use serde::{Deserialize, Serialize};

/// Universal precision formats supported across the Spodeian AI ecosystem
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantizedFormat {
    Fp32,
    Fp16,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Nf4,
    #[default]
    Int8,
    Int4,
    Posit16,
    Posit8,
    Ternary158,
    Log8,
    Bnn1Bit,
}

impl QuantizedFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32 (Single Precision / Studio Baseline)",
            Self::Fp16 => "FP16 (Half Precision / WebGPU)",
            Self::Bf16 => "BF16 (Brain Float 16)",
            Self::Fp8E4M3 => "FP8 E4M3 (Weights)",
            Self::Fp8E5M2 => "FP8 E5M2 (Gradients & Activations)",
            Self::Nf4 => "NF4 (NormalFloat-4 / QLoRA)",
            Self::Int8 => "INT8 (8-Bit Linear Integer)",
            Self::Int4 => "INT4 (4-Bit Packed Integer)",
            Self::Posit16 => "Posit16 <16, 1> (Tapered Unum)",
            Self::Posit8 => "Posit8 <8, 1> (Tapered Unum)",
            Self::Ternary158 => "Ternary 1.58-Bit ({-1, 0, +1} BitNet)",
            Self::Log8 => "Log8 (8-Bit Logarithmic Number System)",
            Self::Bnn1Bit => "BNN 1-Bit ({-1, +1} XNOR)",
        }
    }

    pub fn nominal_bits(self) -> f32 {
        match self {
            Self::Fp32 => 32.0,
            Self::Fp16 | Self::Bf16 | Self::Posit16 => 16.0,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Int8 | Self::Posit8 | Self::Log8 => 8.0,
            Self::Nf4 | Self::Int4 => 4.0,
            Self::Ternary158 => 1.5849625,
            Self::Bnn1Bit => 1.0,
        }
    }
}

/// Evaluation metrics comparing a quantized tensor output against an FP32 baseline
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantizationMetrics {
    pub format: QuantizedFormat,
    pub sqnr_db: f32,
    pub cosine_similarity: f32,
    pub mae: f32,
    pub rmse: f32,
    pub compression_ratio: f32,
}

/// Compute Signal-to-Quantization-Noise Ratio (SQNR in decibels)
pub fn compute_sqnr(quantized: &[f32], reference_fp32: &[f32]) -> f32 {
    if quantized.is_empty() || quantized.len() != reference_fp32.len() {
        return 0.0;
    }

    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;

    for (&q, &r) in quantized.iter().zip(reference_fp32.iter()) {
        signal_power += (r as f64) * (r as f64);
        let diff = (r as f64) - (q as f64);
        noise_power += diff * diff;
    }

    if noise_power <= 1e-15 {
        return 120.0; // Effectively lossless / infinite SQNR capped at 120 dB
    }

    let ratio = signal_power / noise_power;
    if ratio <= 1e-15 {
        return 0.0;
    }

    (10.0 * ratio.log10()) as f32
}

/// Compute Cosine Similarity between two float vectors
pub fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0) as f32
    }
}

/// Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
pub fn compute_mae_rmse(quantized: &[f32], reference: &[f32]) -> (f32, f32) {
    if quantized.is_empty() || quantized.len() != reference.len() {
        return (0.0, 0.0);
    }

    let n = quantized.len() as f64;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;

    for (&q, &r) in quantized.iter().zip(reference.iter()) {
        let diff = ((q as f64) - (r as f64)).abs();
        abs_sum += diff;
        sq_sum += diff * diff;
    }

    ((abs_sum / n) as f32, (sq_sum / n).sqrt() as f32)
}

/// Compute full quantization metrics vs FP32 baseline
pub fn evaluate_quantization_metrics(
    format: QuantizedFormat,
    quantized: &[f32],
    reference_fp32: &[f32],
) -> QuantizationMetrics {
    let sqnr_db = compute_sqnr(quantized, reference_fp32);
    let cosine_similarity = compute_cosine_similarity(quantized, reference_fp32);
    let (mae, rmse) = compute_mae_rmse(quantized, reference_fp32);
    let compression_ratio = 32.0 / format.nominal_bits();

    QuantizationMetrics {
        format,
        sqnr_db,
        cosine_similarity,
        mae,
        rmse,
        compression_ratio,
    }
}

// ============================================================================
// NormalFloat-4 (NF4) Quantiles for Standard Normal Distribution
// ============================================================================

/// 16 NF4 quantiles as defined in QLoRA (Dettmers et al., 2023)
pub static NF4_QUANTILES: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
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

#[inline]
pub fn quantize_to_nf4(val: f32) -> u8 {
    let mut best_idx = 0;
    let mut min_diff = (val - NF4_QUANTILES[0]).abs();

    for i in 1..16 {
        let diff = (val - NF4_QUANTILES[i]).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = i as u8;
        }
    }
    best_idx
}

#[inline]
pub fn dequantize_nf4(code: u8) -> f32 {
    NF4_QUANTILES[(code & 0x0F) as usize]
}

// ============================================================================
// Posit8 <8, 1> Static Lookup Table
// ============================================================================

pub struct PositDecoder;

impl PositDecoder {
    pub fn decode_posit16(bits: u16) -> f32 {
        if bits == 0 {
            return 0.0;
        }
        if bits == 0x8000 {
            return f32::NAN;
        }

        let is_neg = (bits >> 15) == 1;
        let u = if is_neg {
            (!bits).wrapping_add(1)
        } else {
            bits
        };

        let r = (u >> 14) & 1;
        let mut bit_pos = 13i32;
        let mut run_len = 1;

        while bit_pos >= 0 && ((u >> bit_pos) & 1) == r {
            run_len += 1;
            bit_pos -= 1;
        }
        bit_pos -= 1;

        let k = if r == 1 { run_len - 1 } else { -run_len };

        let e = if bit_pos >= 0 {
            let bit = (u >> bit_pos) & 1;
            bit_pos -= 1;
            bit as i32
        } else {
            0
        };

        let frac = if bit_pos >= 0 {
            let num_frac_bits = (bit_pos + 1) as u32;
            let mask = (1 << num_frac_bits) - 1;
            let frac_raw = (u as u32) & mask;
            frac_raw as f32 / (1 << num_frac_bits) as f32
        } else {
            0.0
        };

        let val = (1.0 + frac) * 2.0f32.powi(2 * k + e);
        if is_neg { -val } else { val }
    }

    pub fn get_posit8_lut() -> &'static [f32; 256] {
        static POSIT8_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
        POSIT8_LUT.get_or_init(|| {
            let mut table = [0.0f32; 256];
            for i in 0..256 {
                table[i] = Self::decode_posit16((i as u16) << 8);
            }
            table
        })
    }

    #[inline]
    pub fn decode_posit8(byte: u8) -> f32 {
        Self::get_posit8_lut()[byte as usize]
    }
}

// ============================================================================
// IEEE FP16 & BF16 Encoders/Decoders
// ============================================================================

#[inline]
pub fn encode_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

#[inline]
pub fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn encode_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7C00 | m;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        (sign << 15) | 0x7C00
    } else if new_exp <= 0 {
        if 14 - new_exp > 24 {
            sign << 15
        } else {
            let m = (mant | 0x800000) >> (14 - new_exp + 1);
            (sign << 15) | (m as u16)
        }
    } else {
        let new_mant = (mant >> 13) as u16;
        (sign << 15) | ((new_exp as u16) << 10) | new_mant
    }
}

pub fn decode_fp16(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0x1F {
        if mant != 0 {
            f32::NAN
        } else if sign == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else if exp == 0 {
        if mant == 0 {
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            let val = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else {
        let new_exp = (exp as i32 - 15 + 127) as u32;
        let new_mant = mant << 13;
        f32::from_bits((sign << 31) | (new_exp << 23) | new_mant)
    }
}

// ============================================================================
// FP8 Formats (E4M3 and E5M2)
// ============================================================================

/// Decode FP8 E4M3 (1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits)
pub fn decode_fp8_e4m3(byte: u8) -> f32 {
    let sign = if (byte >> 7) == 1 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;

    if exp == 15 && (byte & 0x07) == 7 {
        return f32::NAN;
    }

    if exp == 0 {
        // Subnormal
        sign * (mant / 8.0) * 2.0f32.powi(-6)
    } else {
        // Normal
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// Decode FP8 E5M2 (1 sign bit, 5 exponent bits with bias 15, 2 mantissa bits)
pub fn decode_fp8_e5m2(byte: u8) -> f32 {
    let sign = if (byte >> 7) == 1 { -1.0 } else { 1.0 };
    let exp = ((byte >> 2) & 0x1F) as i32;
    let mant = (byte & 0x03) as f32;

    if exp == 31 {
        if (byte & 0x03) != 0 {
            f32::NAN
        } else {
            sign * f32::INFINITY
        }
    } else if exp == 0 {
        sign * (mant / 4.0) * 2.0f32.powi(-14)
    } else {
        sign * (1.0 + mant / 4.0) * 2.0f32.powi(exp - 15)
    }
}

// ============================================================================
// SIMD Matrix Multiplications across formats
// ============================================================================

/// 2-bit packed Ternary matrix multiplication ({-1, 0, +1}) with gamma scaling
#[inline]
pub fn ternary_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    gamma: f32,
) {
    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(4);

    for (i, out_val) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let w_offset = i * bytes_per_row;
        let row_bytes = &packed_weights[w_offset..w_offset + bytes_per_row];

        for (b_idx, &byte) in row_bytes.iter().enumerate() {
            let act_offset = b_idx * 4;
            for bit_shift in 0..4 {
                let act_idx = act_offset + bit_shift;
                if act_idx >= in_dim {
                    break;
                }
                let code = (byte >> (bit_shift * 2)) & 0x03;
                let w = match code {
                    0x01 => 1.0f32,
                    0x03 => -1.0f32,
                    _ => 0.0f32,
                };
                sum += w * activations[act_idx];
            }
        }
        *out_val = sum * gamma;
    }
}

/// Vectorized matrix multiplication for INT8 signed integer weights
#[inline]
pub fn int8_matmul_simd_f32(weights: &[i8], activations: &[f32], output: &mut [f32], scale: f32) {
    let in_dim = activations.len();
    assert!(
        weights.len() >= output.len() * in_dim,
        "INT8 weights slice dimensions insufficient"
    );
    let inv_127 = scale / 127.0;

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &weights[w_offset..w_offset + in_dim];
        let mut sum = 0.0f32;

        let iters_4 = in_dim / 4;
        for j in 0..iters_4 {
            let idx = j * 4;
            sum += (w_row[idx] as f32) * activations[idx]
                + (w_row[idx + 1] as f32) * activations[idx + 1]
                + (w_row[idx + 2] as f32) * activations[idx + 2]
                + (w_row[idx + 3] as f32) * activations[idx + 3];
        }

        for j in (iters_4 * 4)..in_dim {
            sum += (w_row[j] as f32) * activations[j];
        }

        *out_val = sum * inv_127;
    }
}

/// Vectorized matrix multiplication for raw Posit8 <8, 1> weights using static L1 LUT
#[inline]
pub fn posit8_matmul_simd_f32(
    raw_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    assert!(
        raw_weights.len() >= output.len() * in_dim,
        "Posit8 weight slice dimensions insufficient"
    );
    let lut = PositDecoder::get_posit8_lut();

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &raw_weights[w_offset..w_offset + in_dim];
        let mut sum = 0.0f32;

        let iters_4 = in_dim / 4;
        for j in 0..iters_4 {
            let idx = j * 4;
            let w0 = lut[w_row[idx] as usize];
            let w1 = lut[w_row[idx + 1] as usize];
            let w2 = lut[w_row[idx + 2] as usize];
            let w3 = lut[w_row[idx + 3] as usize];

            sum += w0 * activations[idx]
                + w1 * activations[idx + 1]
                + w2 * activations[idx + 2]
                + w3 * activations[idx + 3];
        }

        for j in (iters_4 * 4)..in_dim {
            sum += lut[w_row[j] as usize] * activations[j];
        }

        *out_val = sum * scale;
    }
}

/// 4-bit packed integer matrix multiplication (2 weights per byte)
#[inline]
pub fn int4_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(2);
    let inv_7 = scale / 7.0;

    for (i, out_val) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let w_offset = i * bytes_per_row;
        let row_bytes = &packed_weights[w_offset..w_offset + bytes_per_row];

        for (b_idx, &byte) in row_bytes.iter().enumerate() {
            let act_idx0 = b_idx * 2;
            let act_idx1 = act_idx0 + 1;

            let w0 = ((byte & 0x0F) as i8) - 8;
            if act_idx0 < in_dim {
                sum += (w0 as f32) * activations[act_idx0];
            }

            let w1 = (((byte >> 4) & 0x0F) as i8) - 8;
            if act_idx1 < in_dim {
                sum += (w1 as f32) * activations[act_idx1];
            }
        }

        *out_val = sum * inv_7;
    }
}

/// 4-bit NormalFloat (NF4) matrix multiplication
#[inline]
pub fn nf4_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(2);

    for (i, out_val) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let w_offset = i * bytes_per_row;
        let row_bytes = &packed_weights[w_offset..w_offset + bytes_per_row];

        for (b_idx, &byte) in row_bytes.iter().enumerate() {
            let act_idx0 = b_idx * 2;
            let act_idx1 = act_idx0 + 1;

            let w0 = dequantize_nf4(byte & 0x0F);
            if act_idx0 < in_dim {
                sum += w0 * activations[act_idx0];
            }

            let w1 = dequantize_nf4((byte >> 4) & 0x0F);
            if act_idx1 < in_dim {
                sum += w1 * activations[act_idx1];
            }
        }

        *out_val = sum * scale;
    }
}

/// 1-bit Binary Neural Network (BNN) matrix multiplication ({-1, +1} XNOR)
#[inline]
pub fn bnn_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(8);

    for (i, out_val) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let w_offset = i * bytes_per_row;
        let row_bytes = &packed_weights[w_offset..w_offset + bytes_per_row];

        for (b_idx, &byte) in row_bytes.iter().enumerate() {
            let act_base = b_idx * 8;
            for bit in 0..8 {
                let act_idx = act_base + bit;
                if act_idx >= in_dim {
                    break;
                }
                let bit_val = (byte >> bit) & 1;
                let w = if bit_val == 1 { 1.0f32 } else { -1.0f32 };
                sum += w * activations[act_idx];
            }
        }

        *out_val = sum * scale;
    }
}

// ============================================================================
// QuantizedKernel Trait for Uniform Polymorphic Quantization
// ============================================================================

/// Polymorphic quantization kernel trait allowing unified execution & benchmark
pub trait QuantizedKernel: Send + Sync {
    /// Return the specific format represented by this kernel
    fn format(&self) -> QuantizedFormat;

    /// Execute forward matrix multiplication against an activation slice
    fn matmul(&self, activations: &[f32], output: &mut [f32]);

    /// Unpack or dequantize weights into a flat FP32 slice
    fn dequantize(&self, output_weights: &mut [f32]);

    /// Compute SQNR of this quantized kernel relative to an FP32 reference matrix
    fn evaluate_metrics(&self, reference_fp32: &[f32]) -> QuantizationMetrics;
}
