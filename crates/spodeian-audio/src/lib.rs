//! # Spodeian Audio
//!
//! Pure-Rust audio I/O & DSP primitives shared across native and WASM builds:
//! - Lossless RIFF/WAV encoding & decoding (`16-bit PCM`, `24-bit PCM`, `32-bit IEEE float`)
//! - Linear interleaved sample-rate resampling
//! - Acoustic quality metrics (RMS dBFS, peak dBFS, crest factor, DC offset, ZCR)
//! - Zero-allocation in-place soft-clipping limiter

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors raised during audio encoding, decoding, or DSP processing.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("invalid audio parameter: {0}")]
    InvalidParameter(String),
    #[error("malformed WAV container: {0}")]
    MalformedWav(String),
}

/// Supported WAV sample formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcmFormat {
    Pcm16,
    Pcm24,
    Float32,
}

impl PcmFormat {
    pub fn bits_per_sample(self) -> u16 {
        match self {
            Self::Pcm16 => 16,
            Self::Pcm24 => 24,
            Self::Float32 => 32,
        }
    }

    pub fn audio_format_code(self) -> u16 {
        match self {
            Self::Pcm16 | Self::Pcm24 => 1,
            Self::Float32 => 3,
        }
    }
}

/// Decoded interleaved floating-point audio buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

/// Encodes interleaved `[-1.0, 1.0]` float samples into a complete RIFF/WAVE byte buffer.
pub fn encode_wav_bytes(
    samples: &[f32],
    channels: u16,
    sample_rate_hz: u32,
    format: PcmFormat,
) -> Result<Vec<u8>, AudioError> {
    if channels == 0 || sample_rate_hz == 0 {
        return Err(AudioError::InvalidParameter(
            "channels and sample_rate_hz must be non-zero".to_string(),
        ));
    }
    if !samples.len().is_multiple_of(channels as usize) {
        return Err(AudioError::InvalidParameter(
            "samples length must be a multiple of channel count".to_string(),
        ));
    }

    let bits = format.bits_per_sample();
    let bytes_per_sample = (bits / 8) as usize;
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate_hz * u32::from(block_align);
    let data_bytes = (samples.len() * bytes_per_sample) as u32;
    let riff_size = 36 + data_bytes;

    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format.audio_format_code().to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate_hz.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for &s in samples {
        let clamped = if s.is_finite() {
            s.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        match format {
            PcmFormat::Pcm16 => {
                let v = (clamped * 32767.0).round() as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
            PcmFormat::Pcm24 => {
                let v = (clamped * 8_388_607.0).round() as i32;
                let b = v.to_le_bytes();
                out.extend_from_slice(&b[0..3]);
            }
            PcmFormat::Float32 => {
                out.extend_from_slice(&clamped.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Decodes a standard 44-byte-header RIFF/WAVE byte slice (`16-bit`, `24-bit`, or `32-bit float`).
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<AudioBuffer, AudioError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::MalformedWav(
            "missing RIFF/WAVE header".to_string(),
        ));
    }
    let audio_fmt = u16::from_le_bytes([bytes[20], bytes[21]]);
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate_hz = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    let data = &bytes[44..];

    let samples = match (audio_fmt, bits) {
        (1, 16) => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
            .collect(),
        (1, 24) => data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                let sign = if c[2] & 0x80 != 0 { 0xFF } else { 0x00 };
                let raw = i32::from_le_bytes([c[0], c[1], c[2], sign]);
                raw as f32 / 8_388_608.0
            })
            .collect(),
        (3, 32) => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        _ => {
            return Err(AudioError::MalformedWav(format!(
                "unsupported format={audio_fmt} bits={bits}"
            )));
        }
    };

    Ok(AudioBuffer {
        sample_rate_hz,
        channels,
        samples,
    })
}

/// Linearly resamples interleaved audio from `src_rate` to `dst_rate`.
pub fn resample_linear_interleaved(
    samples: &[f32],
    channels: usize,
    src_rate: u32,
    dst_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    if channels == 0 || src_rate == 0 || dst_rate == 0 || !samples.len().is_multiple_of(channels) {
        return Err(AudioError::InvalidParameter(
            "invalid channels or sample rates for resampling".to_string(),
        ));
    }
    if src_rate == dst_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let src_frames = samples.len() / channels;
    let dst_frames = ((src_frames as f64 * dst_rate as f64) / src_rate as f64)
        .round()
        .max(1.0) as usize;
    let mut out = Vec::with_capacity(dst_frames * channels);
    let ratio = src_rate as f64 / dst_rate as f64;
    for df in 0..dst_frames {
        let pos = df as f64 * ratio;
        let idx0 = (pos.floor() as usize).min(src_frames - 1);
        let idx1 = (idx0 + 1).min(src_frames - 1);
        let frac = (pos - idx0 as f64) as f32;
        for ch in 0..channels {
            let s0 = samples[idx0 * channels + ch];
            let s1 = samples[idx1 * channels + ch];
            out.push(s0 + (s1 - s0) * frac);
        }
    }
    Ok(out)
}

/// Summary acoustic quality metrics for a mono or interleaved audio buffer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcousticMetrics {
    pub rms_dbfs: f32,
    pub peak_dbfs: f32,
    pub crest_factor_db: f32,
    pub dc_offset: f32,
}

pub fn compute_acoustic_metrics(samples: &[f32]) -> AcousticMetrics {
    if samples.is_empty() {
        return AcousticMetrics {
            rms_dbfs: -120.0,
            peak_dbfs: -120.0,
            crest_factor_db: 0.0,
            dc_offset: 0.0,
        };
    }
    let mut sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    let mut peak = 0.0_f32;
    for &s in samples {
        let v = if s.is_finite() { s } else { 0.0 };
        sum += v as f64;
        sq_sum += (v as f64) * (v as f64);
        peak = peak.max(v.abs());
    }
    let n = samples.len() as f64;
    let rms = (sq_sum / n).sqrt().max(1e-6) as f32;
    let peak = peak.max(1e-6);
    let rms_dbfs = 20.0 * rms.log10();
    let peak_dbfs = 20.0 * peak.log10();
    AcousticMetrics {
        rms_dbfs,
        peak_dbfs,
        crest_factor_db: peak_dbfs - rms_dbfs,
        dc_offset: (sum / n) as f32,
    }
}

/// Zero-allocation in-place tanh soft-clipping limiter (`drive > 0`), bounded strictly to `[-1.0, 1.0]`.
pub fn soft_clip_tanh_in_place(buffer: &mut [f32], drive: f32) {
    let d = drive.max(1e-4);
    for s in buffer.iter_mut() {
        if s.is_finite() {
            *s = (*s * d).tanh();
        } else {
            *s = 0.0;
        }
    }
}
