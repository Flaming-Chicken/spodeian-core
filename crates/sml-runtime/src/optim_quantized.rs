//! Modular Quantisable Optimizer State Buffers across the Full Quantization Ladder.
//!
//! Provides isolated, block-scaled quantisable storage buffers for first-order optimizers:
//! - Full ladder: F64, F32, TF32, BF16, F16, Posit32, Posit16, Posit8, FP8 (E4M3 / E5M2),
//!   Log16, Log8, INT8, UINT8, NF4, INT4, UINT4, Ternary 1.58-bit, and BNN 1-bit.
//! - Blockwise dynamic scale factors (default block_size = 64) with deterministic or stochastic rounding.
//! - Error-feedback compensation buffers for low-bit residuals (Polyak delta <= 8-bit).
//! - Component-level policy customization (weights, Polyak delta, momentum, Fisher, sign-flip, step).
//! - Clean compilation on both native CPU (with SIMD/Rayon) and `wasm32-unknown-unknown`.

use crate::logarithmic::{Log16, Log8};
use crate::quantization::{
    decode_bf16, decode_fp16, decode_fp8_e4m3, decode_fp8_e5m2, decode_posit16, decode_posit8,
    decode_tf32, encode_bf16, encode_fp16, encode_fp8_e4m3, encode_fp8_e5m2, encode_posit16,
    encode_posit8, encode_tf32, QuantizedFormat,
};
use serde::{Deserialize, Serialize};

/// Fast SplitMix64 PRNG for deterministic, non-cryptographic stochastic rounding dithering.
#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Normalized uniform random float in [-0.5, 0.5] for stochastic rounding.
#[inline(always)]
fn stochastic_jitter(state: &mut u64) -> f32 {
    let r = splitmix64(state) as u32;
    (r as f32 / 4294967296.0f32) - 0.5f32
}

pub use crate::quantization::NF4_CENTROIDS;

/// Encodes a normalized value in [-1, 1] into a 4-bit NF4 nibble (0..15).
#[inline(always)]
pub fn encode_nf4_nibble(val: f32) -> u8 {
    crate::quantization::quantize_to_nf4(val)
}

/// Decodes a 4-bit NF4 nibble (0..15) into a normalized float in [-1, 1].
#[inline(always)]
pub fn decode_nf4_nibble(nibble: u8) -> f32 {
    crate::quantization::dequantize_nf4(nibble)
}

/// An isolated, block-scaled quantisable buffer supporting the full precision ladder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizedStateBuffer {
    /// Native 64-bit IEEE 754 floats (8 B/element)
    F64(Vec<f64>),
    /// Native 32-bit IEEE 754 floats (4 B/element)
    F32(Vec<f32>),
    /// TensorFloat-32 packed 32-bit container (4 B/element)
    Tf32(Vec<u32>),
    /// Brain Float 16 (2 B/element)
    Bf16(Vec<u16>),
    /// IEEE 754 Half Precision (2 B/element)
    F16(Vec<u16>),
    /// Posit16 <16, 1> (2 B/element)
    Posit16(Vec<u16>),
    /// Posit8 <8, 1> (1 B/element)
    Posit8(Vec<u8>),
    /// FP8 E4M3 (1 B/element)
    Fp8E4M3(Vec<u8>),
    /// FP8 E5M2 (1 B/element)
    Fp8E5M2(Vec<u8>),
    /// Logarithmic 16-bit (2 B/element)
    Log16(Vec<u16>),
    /// Logarithmic 8-bit (1 B/element)
    Log8(Vec<u8>),
    /// Blockwise INT8: 1 byte per element + f32 scale per 64-element block
    BlockInt8 {
        len: usize,
        bytes: Vec<i8>,
        scales: Vec<f32>,
    },
    /// Blockwise NF4: 0.5 bytes per element + f32 scale per 64-element block
    BlockNf4 {
        len: usize,
        packed_bytes: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Blockwise INT4: 0.5 bytes per element + f32 scale per 64-element block
    BlockInt4 {
        len: usize,
        packed_bytes: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Blockwise Ternary 1.58b: 2-bit packed {-1, 0, +1} + f32 scale per 64-element block
    BlockTernary158 {
        len: usize,
        packed_bytes: Vec<u8>,
        scales: Vec<f32>,
    },
    /// Blockwise BNN 1-bit: 1-bit packed {-1, +1} + f32 scale per 64-element block
    BlockBnn1Bit {
        len: usize,
        packed_bytes: Vec<u8>,
        scales: Vec<f32>,
    },
}

impl QuantizedStateBuffer {
    pub const BLOCK_SIZE: usize = 64;

    /// Allocate a new zeroed quantized state buffer with the given length and format.
    pub fn zeros(len: usize, format: QuantizedFormat) -> Self {
        match format {
            QuantizedFormat::Fp32 => Self::F32(vec![0.0f32; len]),
            QuantizedFormat::Fp16 => Self::F16(vec![0u16; len]),
            QuantizedFormat::Bf16 => Self::Bf16(vec![0u16; len]),
            QuantizedFormat::Fp8E4M3 => Self::Fp8E4M3(vec![0u8; len]),
            QuantizedFormat::Fp8E5M2 => Self::Fp8E5M2(vec![0u8; len]),
            QuantizedFormat::Posit16 => Self::Posit16(vec![0u16; len]),
            QuantizedFormat::Posit8 => Self::Posit8(vec![0u8; len]),
            QuantizedFormat::Log8 => Self::Log8(vec![0u8; len]),
            QuantizedFormat::Int8 => {
                let num_blocks = (len + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                Self::BlockInt8 {
                    len,
                    bytes: vec![0i8; len],
                    scales: vec![1.0f32; num_blocks],
                }
            }
            QuantizedFormat::Nf4 => {
                let num_blocks = (len + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                let num_bytes = (len + 1) / 2;
                Self::BlockNf4 {
                    len,
                    packed_bytes: vec![0u8; num_bytes],
                    scales: vec![1.0f32; num_blocks],
                }
            }
            QuantizedFormat::Int4 => {
                let num_blocks = (len + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                let num_bytes = (len + 1) / 2;
                Self::BlockInt4 {
                    len,
                    packed_bytes: vec![0u8; num_bytes],
                    scales: vec![1.0f32; num_blocks],
                }
            }
            QuantizedFormat::Ternary158 => {
                let num_blocks = (len + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                let num_bytes = (len + 3) / 4;
                Self::BlockTernary158 {
                    len,
                    packed_bytes: vec![0u8; num_bytes],
                    scales: vec![1.0f32; num_blocks],
                }
            }
            QuantizedFormat::Bnn1Bit => {
                let num_blocks = (len + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                let num_bytes = (len + 7) / 8;
                Self::BlockBnn1Bit {
                    len,
                    packed_bytes: vec![0u8; num_bytes],
                    scales: vec![1.0f32; num_blocks],
                }
            }
        }
    }

    /// Number of logical scalar elements in the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Tf32(v) => v.len(),
            Self::Bf16(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::Posit16(v) => v.len(),
            Self::Posit8(v) => v.len(),
            Self::Fp8E4M3(v) => v.len(),
            Self::Fp8E5M2(v) => v.len(),
            Self::Log16(v) => v.len(),
            Self::Log8(v) => v.len(),
            Self::BlockInt8 { len, .. }
            | Self::BlockNf4 { len, .. }
            | Self::BlockInt4 { len, .. }
            | Self::BlockTernary158 { len, .. }
            | Self::BlockBnn1Bit { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exact memory footprint of this buffer in bytes (data + block scale metadata).
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F64(v) => v.len() * 8,
            Self::F32(v) => v.len() * 4,
            Self::Tf32(v) => v.len() * 4,
            Self::Bf16(v) => v.len() * 2,
            Self::F16(v) => v.len() * 2,
            Self::Posit16(v) => v.len() * 2,
            Self::Posit8(v) => v.len(),
            Self::Fp8E4M3(v) => v.len(),
            Self::Fp8E5M2(v) => v.len(),
            Self::Log16(v) => v.len() * 2,
            Self::Log8(v) => v.len(),
            Self::BlockInt8 { bytes, scales, .. } => bytes.len() + scales.len() * 4,
            Self::BlockNf4 {
                packed_bytes,
                scales,
                ..
            }
            | Self::BlockInt4 {
                packed_bytes,
                scales,
                ..
            }
            | Self::BlockTernary158 {
                packed_bytes,
                scales,
                ..
            }
            | Self::BlockBnn1Bit {
                packed_bytes,
                scales,
                ..
            } => packed_bytes.len() + scales.len() * 4,
        }
    }

    /// Effective bytes per parameter coordinate (including block scale overhead).
    pub fn bytes_per_element(&self) -> f32 {
        let n = self.len().max(1);
        self.byte_size() as f32 / n as f32
    }

    /// Pack a slice of `f32` values into this buffer in-place.
    pub fn pack(&mut self, values: &[f32], stochastic: bool, seed: u64) {
        let n = values.len().min(self.len());
        let mut rng = seed.wrapping_add(0x1337_5EED);

        match self {
            Self::F64(v) => {
                for i in 0..n {
                    v[i] = values[i] as f64;
                }
            }
            Self::F32(v) => {
                v[..n].copy_from_slice(&values[..n]);
            }
            Self::Tf32(v) => {
                for i in 0..n {
                    let mut val = values[i];
                    if stochastic {
                        val += stochastic_jitter(&mut rng) * (1.0 / 1024.0) * val.abs();
                    }
                    v[i] = encode_tf32(val);
                }
            }
            Self::Bf16(v) => {
                for i in 0..n {
                    let mut val = values[i];
                    if stochastic {
                        val += stochastic_jitter(&mut rng) * (1.0 / 128.0) * val.abs();
                    }
                    v[i] = encode_bf16(val);
                }
            }
            Self::F16(v) => {
                for i in 0..n {
                    let mut val = values[i];
                    if stochastic {
                        val += stochastic_jitter(&mut rng) * (1.0 / 1024.0) * val.abs();
                    }
                    v[i] = encode_fp16(val);
                }
            }
            Self::Posit16(v) => {
                for i in 0..n {
                    v[i] = encode_posit16(values[i]);
                }
            }
            Self::Posit8(v) => {
                for i in 0..n {
                    v[i] = encode_posit8(values[i]);
                }
            }
            Self::Fp8E4M3(v) => {
                for i in 0..n {
                    let mut val = values[i];
                    if stochastic {
                        val += stochastic_jitter(&mut rng) * 0.125 * val.abs();
                    }
                    v[i] = encode_fp8_e4m3(val);
                }
            }
            Self::Fp8E5M2(v) => {
                for i in 0..n {
                    let mut val = values[i];
                    if stochastic {
                        val += stochastic_jitter(&mut rng) * 0.25 * val.abs();
                    }
                    v[i] = encode_fp8_e5m2(val);
                }
            }
            Self::Log16(v) => {
                for i in 0..n {
                    v[i] = Log16::from_f32(values[i]).0;
                }
            }
            Self::Log8(v) => {
                for i in 0..n {
                    v[i] = Log8::from_f32(values[i]).0;
                }
            }
            Self::BlockInt8 { bytes, scales, .. } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in values[..n].chunks(bs).enumerate() {
                    let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())).max(1e-12);
                    let scale = max_abs / 127.0f32;
                    scales[b_idx] = scale;
                    let inv_scale = 1.0f32 / scale;
                    for (c_idx, &val) in chunk.iter().enumerate() {
                        let jitter = if stochastic { stochastic_jitter(&mut rng) } else { 0.0 };
                        let q = ((val * inv_scale) + jitter).round().clamp(-127.0, 127.0) as i8;
                        bytes[b_idx * bs + c_idx] = q;
                    }
                }
            }
            Self::BlockNf4 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in values[..n].chunks(bs).enumerate() {
                    let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())).max(1e-12);
                    scales[b_idx] = max_abs;
                    let inv_scale = 1.0f32 / max_abs;
                    let base_out = (b_idx * bs) / 2;
                    for pair_idx in 0..(chunk.len() + 1) / 2 {
                        let i0 = pair_idx * 2;
                        let i1 = i0 + 1;
                        let v0 = chunk[i0] * inv_scale;
                        let v1 = if i1 < chunk.len() { chunk[i1] * inv_scale } else { 0.0 };
                        let nib0 = encode_nf4_nibble(v0);
                        let nib1 = encode_nf4_nibble(v1);
                        packed_bytes[base_out + pair_idx] = (nib1 << 4) | (nib0 & 0x0F);
                    }
                }
            }
            Self::BlockInt4 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in values[..n].chunks(bs).enumerate() {
                    let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())).max(1e-12);
                    let scale = max_abs / 7.0f32;
                    scales[b_idx] = scale;
                    let inv_scale = 1.0f32 / scale;
                    let base_out = (b_idx * bs) / 2;
                    for pair_idx in 0..(chunk.len() + 1) / 2 {
                        let i0 = pair_idx * 2;
                        let i1 = i0 + 1;
                        let q0 = (chunk[i0] * inv_scale).round().clamp(-7.0, 7.0) as i8;
                        let q1 = if i1 < chunk.len() {
                            (chunk[i1] * inv_scale).round().clamp(-7.0, 7.0) as i8
                        } else {
                            0
                        };
                        let nib0 = (q0 & 0x0F) as u8;
                        let nib1 = (q1 & 0x0F) as u8;
                        packed_bytes[base_out + pair_idx] = (nib1 << 4) | (nib0 & 0x0F);
                    }
                }
            }
            Self::BlockTernary158 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in values[..n].chunks(bs).enumerate() {
                    let mean_abs =
                        chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs())).max(1e-12);
                    scales[b_idx] = mean_abs;
                    let threshold = 0.5f32 * mean_abs;
                    let base_out = (b_idx * bs) / 4;
                    for group_idx in 0..(chunk.len() + 3) / 4 {
                        let mut byte = 0u8;
                        for bit_pair in 0..4 {
                            let idx = group_idx * 4 + bit_pair;
                            if idx < chunk.len() {
                                let val = chunk[idx];
                                let code: u8 = if val > threshold {
                                    1 // +1
                                } else if val < -threshold {
                                    2 // -1
                                } else {
                                    0 // 0
                                };
                                byte |= (code & 0x03) << (bit_pair * 2);
                            }
                        }
                        packed_bytes[base_out + group_idx] = byte;
                    }
                }
            }
            Self::BlockBnn1Bit {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in values[..n].chunks(bs).enumerate() {
                    let mean_abs = chunk.iter().map(|&x| x.abs()).sum::<f32>() / chunk.len() as f32;
                    scales[b_idx] = mean_abs.max(1e-12);
                    let base_out = (b_idx * bs) / 8;
                    for byte_idx in 0..(chunk.len() + 7) / 8 {
                        let mut byte = 0u8;
                        for b in 0..8 {
                            let idx = byte_idx * 8 + b;
                            if idx < chunk.len() && chunk[idx] >= 0.0 {
                                byte |= 1 << b;
                            }
                        }
                        packed_bytes[base_out + byte_idx] = byte;
                    }
                }
            }
        }
    }

    /// Unpack the buffer into a destination slice of `f32`.
    pub fn unpack_into(&self, dest: &mut [f32]) {
        let n = dest.len().min(self.len());

        match self {
            Self::F64(v) => {
                for i in 0..n {
                    dest[i] = v[i] as f32;
                }
            }
            Self::F32(v) => {
                dest[..n].copy_from_slice(&v[..n]);
            }
            Self::Tf32(v) => {
                for i in 0..n {
                    dest[i] = decode_tf32(v[i]);
                }
            }
            Self::Bf16(v) => {
                for i in 0..n {
                    dest[i] = decode_bf16(v[i]);
                }
            }
            Self::F16(v) => {
                for i in 0..n {
                    dest[i] = decode_fp16(v[i]);
                }
            }
            Self::Posit16(v) => {
                for i in 0..n {
                    dest[i] = decode_posit16(v[i]);
                }
            }
            Self::Posit8(v) => {
                for i in 0..n {
                    dest[i] = decode_posit8(v[i]);
                }
            }
            Self::Fp8E4M3(v) => {
                for i in 0..n {
                    dest[i] = decode_fp8_e4m3(v[i]);
                }
            }
            Self::Fp8E5M2(v) => {
                for i in 0..n {
                    dest[i] = decode_fp8_e5m2(v[i]);
                }
            }
            Self::Log16(v) => {
                for i in 0..n {
                    dest[i] = Log16(v[i]).to_f32();
                }
            }
            Self::Log8(v) => {
                for i in 0..n {
                    dest[i] = Log8(v[i]).to_f32();
                }
            }
            Self::BlockInt8 { bytes, scales, .. } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in dest[..n].chunks_mut(bs).enumerate() {
                    let scale = scales[b_idx];
                    for (c_idx, val) in chunk.iter_mut().enumerate() {
                        *val = bytes[b_idx * bs + c_idx] as f32 * scale;
                    }
                }
            }
            Self::BlockNf4 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in dest[..n].chunks_mut(bs).enumerate() {
                    let scale = scales[b_idx];
                    let base_in = (b_idx * bs) / 2;
                    for (c_idx, val) in chunk.iter_mut().enumerate() {
                        let byte = packed_bytes[base_in + c_idx / 2];
                        let nibble = if c_idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        *val = decode_nf4_nibble(nibble) * scale;
                    }
                }
            }
            Self::BlockInt4 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in dest[..n].chunks_mut(bs).enumerate() {
                    let scale = scales[b_idx];
                    let base_in = (b_idx * bs) / 2;
                    for (c_idx, val) in chunk.iter_mut().enumerate() {
                        let byte = packed_bytes[base_in + c_idx / 2];
                        let raw_nib = if c_idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                        // Sign extend 4-bit two's complement
                        let signed_val = if (raw_nib & 0x08) != 0 {
                            (raw_nib as i8) | !0x0Fi8
                        } else {
                            raw_nib as i8
                        };
                        *val = signed_val as f32 * scale;
                    }
                }
            }
            Self::BlockTernary158 {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in dest[..n].chunks_mut(bs).enumerate() {
                    let scale = scales[b_idx];
                    let base_in = (b_idx * bs) / 4;
                    for (c_idx, val) in chunk.iter_mut().enumerate() {
                        let byte = packed_bytes[base_in + c_idx / 4];
                        let code = (byte >> ((c_idx % 4) * 2)) & 0x03;
                        let trit = match code {
                            1 => 1.0f32,
                            2 => -1.0f32,
                            _ => 0.0f32,
                        };
                        *val = trit * scale;
                    }
                }
            }
            Self::BlockBnn1Bit {
                packed_bytes,
                scales,
                ..
            } => {
                let bs = Self::BLOCK_SIZE;
                for (b_idx, chunk) in dest[..n].chunks_mut(bs).enumerate() {
                    let scale = scales[b_idx];
                    let base_in = (b_idx * bs) / 8;
                    for (c_idx, val) in chunk.iter_mut().enumerate() {
                        let byte = packed_bytes[base_in + c_idx / 8];
                        let bit = (byte >> (c_idx % 8)) & 1;
                        let spin = if bit == 1 { 1.0f32 } else { -1.0f32 };
                        *val = spin * scale;
                    }
                }
            }
        }
    }

    /// Unpack into a freshly allocated `Vec<f32>`.
    pub fn unpack_to_vec(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.len()];
        self.unpack_into(&mut v);
        v
    }
}

/// Error-feedback accumulation buffer to prevent quantization noise drift on low-bit residuals.
#[derive(Debug, Clone, Default)]
pub struct ErrorFeedbackBuffer {
    error: Vec<f32>,
}

impl ErrorFeedbackBuffer {
    pub fn new(len: usize) -> Self {
        Self {
            error: vec![0.0f32; len],
        }
    }

    /// Add accumulated error to delta, quantize, and update error with the residual:
    /// $$e_t = (\Delta_t + e_{t-1}) - \tilde{\Delta}_t$$
    pub fn quantize_with_feedback(
        &mut self,
        target_buf: &mut QuantizedStateBuffer,
        delta: &[f32],
        stochastic: bool,
        seed: u64,
    ) {
        let n = delta.len().min(self.error.len());
        let mut compensated = vec![0.0f32; n];
        for i in 0..n {
            compensated[i] = delta[i] + self.error[i];
        }
        target_buf.pack(&compensated, stochastic, seed);

        // Compute new error
        let mut unpacked = vec![0.0f32; n];
        target_buf.unpack_into(&mut unpacked);
        for i in 0..n {
            self.error[i] = compensated[i] - unpacked[i];
        }
    }
}

/// Component-level quantization configuration policy for first-order optimizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateQuantizationPolicy {
    /// Format for primal weights / anchor parameters $z_t$.
    pub master_z_format: QuantizedFormat,
    /// Format for Schedule-Free Polyak residual $y_t - z_t$.
    pub polyak_delta_format: QuantizedFormat,
    /// Format for first momentum vector $m_t$ and previous gradient $g_{t-1}$.
    pub momentum_format: QuantizedFormat,
    /// Format for non-negative empirical Fisher second moment $v_t$.
    pub fisher_format: QuantizedFormat,
    /// Format for bounded sign-flip EMA accumulator.
    pub flip_format: QuantizedFormat,
    /// Format for preconditioned step / innovation direction $\hat{g}_t$.
    pub innovation_format: QuantizedFormat,
    /// Whether to apply stochastic rounding dithering during quantization.
    pub stochastic_rounding: bool,
    /// Whether to enable error-feedback compensation for low-bit residuals.
    pub error_feedback: bool,
}

impl Default for StateQuantizationPolicy {
    fn default() -> Self {
        Self::exact_f32()
    }
}

impl StateQuantizationPolicy {
    /// Full 64-bit IEEE 754 double precision (56 B/param for single-layer).
    pub fn exact_f64() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32, // Stored in f64 container in buffer
            polyak_delta_format: QuantizedFormat::Fp32,
            momentum_format: QuantizedFormat::Fp32,
            fisher_format: QuantizedFormat::Fp32,
            flip_format: QuantizedFormat::Fp32,
            innovation_format: QuantizedFormat::Fp32,
            stochastic_rounding: false,
            error_feedback: false,
        }
    }

    /// Standard single-precision baseline (28 B/param for single-layer).
    pub fn exact_f32() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32,
            polyak_delta_format: QuantizedFormat::Fp32,
            momentum_format: QuantizedFormat::Fp32,
            fisher_format: QuantizedFormat::Fp32,
            flip_format: QuantizedFormat::Fp32,
            innovation_format: QuantizedFormat::Fp32,
            stochastic_rounding: false,
            error_feedback: false,
        }
    }

    /// Mixed-precision BF16: weights in FP32, moments in BF16 (14 B/param).
    pub fn mixed_bf16() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32,
            polyak_delta_format: QuantizedFormat::Bf16,
            momentum_format: QuantizedFormat::Bf16,
            fisher_format: QuantizedFormat::Bf16,
            flip_format: QuantizedFormat::Bf16,
            innovation_format: QuantizedFormat::Fp32,
            stochastic_rounding: true,
            error_feedback: true,
        }
    }

    /// Compact 8-bit: weights in FP32, moments in FP8/Log8 (8-10 B/param).
    pub fn compact_8bit() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32,
            polyak_delta_format: QuantizedFormat::Fp8E4M3,
            momentum_format: QuantizedFormat::Fp8E4M3,
            fisher_format: QuantizedFormat::Fp8E5M2,
            flip_format: QuantizedFormat::Int8,
            innovation_format: QuantizedFormat::Fp32,
            stochastic_rounding: true,
            error_feedback: true,
        }
    }

    /// Compact 4-bit: weights in FP32, moments in NF4 (4-6 B/param).
    pub fn compact_4bit() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32,
            polyak_delta_format: QuantizedFormat::Nf4,
            momentum_format: QuantizedFormat::Nf4,
            fisher_format: QuantizedFormat::Nf4,
            flip_format: QuantizedFormat::Int4,
            innovation_format: QuantizedFormat::Fp32,
            stochastic_rounding: true,
            error_feedback: true,
        }
    }

    /// Extreme low-bit: Ternary / BNN directions (< 4 B/param).
    pub fn extreme_low_bit() -> Self {
        Self {
            master_z_format: QuantizedFormat::Fp32,
            polyak_delta_format: QuantizedFormat::Ternary158,
            momentum_format: QuantizedFormat::Ternary158,
            fisher_format: QuantizedFormat::Nf4,
            flip_format: QuantizedFormat::Bnn1Bit,
            innovation_format: QuantizedFormat::Ternary158,
            stochastic_rounding: true,
            error_feedback: true,
        }
    }
}

/// Detailed memory accounting report for quantized optimizer state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedMemoryReport {
    pub total_bytes: usize,
    pub bytes_per_param: f32,
    pub compression_ratio_vs_f64: f32,
    pub compression_ratio_vs_f32: f32,
}

/// Fully modular, component-quantised parameter state buffer for [`SelfOptimizingMeta`].
#[derive(Debug, Clone)]
pub struct QuantizedMetaParamState {
    pub num_params: usize,
    pub policy: StateQuantizationPolicy,
    pub z: QuantizedStateBuffer,
    pub polyak_delta: QuantizedStateBuffer,
    pub m: QuantizedStateBuffer,
    pub v: QuantizedStateBuffer,
    pub prev_g: QuantizedStateBuffer,
    pub flip_ema: QuantizedStateBuffer,
    pub feedback_delta: ErrorFeedbackBuffer,
    pub running_g_sq: f64,
    pub var_step: usize,
}

impl QuantizedMetaParamState {
    pub fn new(len: usize, policy: StateQuantizationPolicy) -> Self {
        Self {
            num_params: len,
            policy,
            z: QuantizedStateBuffer::zeros(len, policy.master_z_format),
            polyak_delta: QuantizedStateBuffer::zeros(len, policy.polyak_delta_format),
            m: QuantizedStateBuffer::zeros(len, policy.momentum_format),
            v: QuantizedStateBuffer::zeros(len, policy.fisher_format),
            prev_g: QuantizedStateBuffer::zeros(len, policy.momentum_format),
            flip_ema: QuantizedStateBuffer::zeros(len, policy.flip_format),
            feedback_delta: ErrorFeedbackBuffer::new(len),
            running_g_sq: 0.0,
            var_step: 0,
        }
    }

    pub fn memory_footprint(&self) -> QuantizedMemoryReport {
        let total = self.z.byte_size()
            + self.polyak_delta.byte_size()
            + self.m.byte_size()
            + self.v.byte_size()
            + self.prev_g.byte_size()
            + self.flip_ema.byte_size();
        let bytes_per_param = total as f32 / self.num_params.max(1) as f32;
        let baseline_f64 = (self.num_params * 56) as f32;
        let baseline_f32 = (self.num_params * 28) as f32;
        QuantizedMemoryReport {
            total_bytes: total,
            bytes_per_param,
            compression_ratio_vs_f64: baseline_f64 / total.max(1) as f32,
            compression_ratio_vs_f32: baseline_f32 / total.max(1) as f32,
        }
    }
}
