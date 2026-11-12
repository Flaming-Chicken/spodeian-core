//! Universal serialization, columnar table parsing, and multi-format export engine.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides the canonical cross-project serialization substrate:
//! - **CSV / TSV**: RFC 4180-compliant quote escaping, line parsing, and table streaming.
//! - **JSON**: Pretty and compact JSON document export and import.
//! - **BSON**: Compressed BSON binary serialization with Zlib compression.
//! - **SafeTensors**: Tensor weight checkpoints and golden parity verification.
//! - **Audio WAV**: Pure-Rust IEEE 32-bit float PCM RIFF/WAVE audio serialization.

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Error, Debug)]
pub enum ExportError {
    #[cfg(feature = "json")]
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[cfg(feature = "bson")]
    #[error("BSON serialization error: {0}")]
    BsonSer(#[from] bson::ser::Error),

    #[cfg(feature = "bson")]
    #[error("BSON deserialization error: {0}")]
    BsonDe(#[from] bson::de::Error),

    #[cfg(feature = "safetensors")]
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("CSV parse error: {0}")]
    Csv(String),

    #[error("Audio format error: {0}")]
    Audio(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ============================================================================
// 1. CSV & TSV Streaming & Escaping
// ============================================================================

/// Escapes a string field for CSV inclusion following standard RFC 4180 rules.
///
/// # Examples
///
/// ```rust
/// use spodeian_export::escape_csv_field;
///
/// assert_eq!(escape_csv_field("simple"), "simple");
/// assert_eq!(escape_csv_field("hello, world"), "\"hello, world\"");
/// assert_eq!(escape_csv_field("quote: \"hi\""), "\"quote: \"\"hi\"\"\"");
/// ```
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}

/// Parses a single CSV line into discrete fields respecting quoted delimiters and escaped quotes.
///
/// # Examples
///
/// ```rust
/// use spodeian_export::parse_csv_line;
///
/// let fields = parse_csv_line("AAPL,150.25,\"New York, USA\"");
/// assert_eq!(fields, vec!["AAPL", "150.25", "New York, USA"]);
/// ```
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' => {
                in_quotes = true;
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Zero-copy optimized CSV line parsing returning borrowed slices whenever unquoted.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn parse_csv_line_borrowed<'a>(line: &'a str) -> Vec<std::borrow::Cow<'a, str>> {
    if !line.contains('"') {
        return line
            .split(',')
            .map(|s| std::borrow::Cow::Borrowed(s.trim()))
            .collect();
    }

    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' => {
                in_quotes = true;
            }
            ',' if !in_quotes => {
                fields.push(std::borrow::Cow::Owned(current.trim().to_string()));
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    fields.push(std::borrow::Cow::Owned(current.trim().to_string()));
    fields
}

/// Serializes headers and tabular string rows into standard RFC 4180 CSV text.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_csv(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    let header_line: Vec<String> = headers.iter().map(|&h| escape_csv_field(h)).collect();
    out.push_str(&header_line.join(","));
    out.push('\n');

    for row in rows {
        let line: Vec<String> = row.iter().map(|f| escape_csv_field(f)).collect();
        out.push_str(&line.join(","));
        out.push('\n');
    }
    out
}

/// Parses an entire CSV document into headers and rows.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn parse_csv_table(csv_content: &str) -> Result<(Vec<String>, Vec<Vec<String>>), ExportError> {
    let mut lines = csv_content.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines
        .next()
        .ok_or_else(|| ExportError::Csv("Empty CSV input".to_string()))?;
    let headers = parse_csv_line(header_line);

    let mut rows = Vec::new();
    for line in lines {
        rows.push(parse_csv_line(line));
    }
    Ok((headers, rows))
}

/// Serializes headers and tabular rows into Tab-Separated Values (TSV).
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_tsv(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str(&headers.join("\t"));
    out.push('\n');

    for row in rows {
        out.push_str(&row.join("\t"));
        out.push('\n');
    }
    out
}

// ============================================================================
// 2. JSON Serialization
// ============================================================================

#[cfg(feature = "json")]
/// Exports any serializable data structure into formatted JSON string.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_json<T: Serialize>(data: &T) -> Result<String, ExportError> {
    serde_json::to_string_pretty(data).map_err(ExportError::Json)
}

#[cfg(feature = "json")]
/// Imports any deserializable data structure from a JSON string.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn import_from_json<T: DeserializeOwned>(json_str: &str) -> Result<T, ExportError> {
    serde_json::from_str(json_str).map_err(ExportError::Json)
}

// ============================================================================
// 3. Compressed BSON Binary Serialization
// ============================================================================

#[cfg(feature = "bson")]
/// Exports any serializable data structure into compressed BSON binary bytes (Zlib-compressed BSON).
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_compressed_bson<T: Serialize>(data: &T) -> Result<Vec<u8>, ExportError> {
    let bson_bytes = bson::to_vec(data).map_err(ExportError::BsonSer)?;
    Ok(miniz_oxide::deflate::compress_to_vec_zlib(&bson_bytes, 6))
}

#[cfg(feature = "bson")]
/// Imports and restores any deserializable data structure from a compressed (or raw) BSON slice.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn import_from_compressed_bson<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ExportError> {
    let bson_bytes =
        miniz_oxide::inflate::decompress_to_vec_zlib(bytes).unwrap_or_else(|_| bytes.to_vec());
    bson::from_slice::<T>(&bson_bytes).map_err(ExportError::BsonDe)
}

// ============================================================================
// 4. SafeTensors Checkpoint Serialization (Neural Weights & Parity Golden Records)
// ============================================================================

#[cfg(feature = "safetensors")]
/// Serializes a map of named floating-point tensors into a pure .safetensors binary buffer.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_tensors_to_safetensors(
    tensors: &[(&str, &[usize], &[f32])],
) -> Result<Vec<u8>, ExportError> {
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::BTreeMap;

    let mut byte_buffers: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    for (_, _, data) in tensors {
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &val in *data {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        byte_buffers.push(buf);
    }

    let mut views = BTreeMap::new();
    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        let view = TensorView::new(Dtype::F32, shape.to_vec(), &byte_buffers[i])
            .map_err(|e| ExportError::SafeTensors(format!("{e}")))?;
        views.insert(name.to_string(), view);
    }

    safetensors::serialize(&views, &None).map_err(|e| ExportError::SafeTensors(format!("{e}")))
}

#[cfg(feature = "safetensors")]
/// Deserializes a .safetensors buffer into a named map of (shape, float-data).
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn import_tensors_from_safetensors(
    bytes: &[u8],
) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>, ExportError> {
    use safetensors::SafeTensors;

    let st =
        SafeTensors::deserialize(bytes).map_err(|e| ExportError::SafeTensors(format!("{e}")))?;

    let mut result = Vec::new();
    for (name, view) in st.tensors() {
        let raw_data = view.data();
        let float_count = raw_data.len() / core::mem::size_of::<f32>();
        let mut float_vec = Vec::with_capacity(float_count);
        for chunk in raw_data.chunks_exact(4) {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            float_vec.push(val);
        }
        result.push((name, view.shape().to_vec(), float_vec));
    }
    Ok(result)
}

// ============================================================================
// 5. Pure-Rust IEEE 32-bit Float PCM WAV Audio Export
// ============================================================================

/// Serializes raw 32-bit float audio samples into a standard RIFF/WAVE file buffer.
///
/// Uses format tag `0x0003` (IEEE Float) with full cross-platform WASM compatibility.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_wav_pcm32f(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>, ExportError> {
    if channels == 0 {
        return Err(ExportError::Audio("Channel count must be > 0".to_string()));
    }

    let bytes_per_sample = 4u32;
    let block_align = (channels as u32) * bytes_per_sample;
    let byte_rate = sample_rate * block_align;
    let data_bytes = (samples.len() as u32) * bytes_per_sample;
    let riff_chunk_size = 36 + data_bytes;

    let mut wav = Vec::with_capacity((44 + data_bytes) as usize);

    // 1. RIFF Header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // 2. fmt Sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size = 16 for PCM
    wav.extend_from_slice(&3u16.to_le_bytes()); // AudioFormat = 3 (IEEE Float)
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&(block_align as u16).to_le_bytes());
    wav.extend_from_slice(&32u16.to_le_bytes()); // BitsPerSample = 32

    // 3. data Sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    Ok(wav)
}

/// Pure-Rust audio serialization into standard 16-bit signed integer linear PCM RIFF/WAVE format.
///
/// Uses format tag `0x0001` (Standard Linear PCM), 16-bit samples, compatible with all browsers and media players.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn export_to_wav_pcm16i(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>, ExportError> {
    if channels == 0 {
        return Err(ExportError::Audio("Channel count must be > 0".to_string()));
    }

    let bytes_per_sample = 2u32;
    let block_align = (channels as u32) * bytes_per_sample;
    let byte_rate = sample_rate * block_align;
    let data_bytes = (samples.len() as u32) * bytes_per_sample;
    let riff_chunk_size = 36 + data_bytes;

    let mut wav = Vec::with_capacity((44 + data_bytes) as usize);

    // 1. RIFF Header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // 2. fmt Sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size = 16 for PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat = 1 (Linear PCM)
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&(block_align as u16).to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample = 16

    // 3. data Sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let s_i16 = (clamped * 32767.0).round() as i16;
        wav.extend_from_slice(&s_i16.to_le_bytes());
    }

    Ok(wav)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_table_roundtrip() {
        let headers = vec!["symbol", "price", "notes"];
        let rows = vec![
            vec![
                "AAPL".to_string(),
                "150.25".to_string(),
                "Standard".to_string(),
            ],
            vec![
                "GOOG".to_string(),
                "2800.50".to_string(),
                "Contains, comma and \"quotes\"".to_string(),
            ],
        ];

        let csv_text = export_to_csv(&headers, &rows);
        let (parsed_headers, parsed_rows) = parse_csv_table(&csv_text).expect("CSV parsed");

        assert_eq!(parsed_headers, vec!["symbol", "price", "notes"]);
        assert_eq!(parsed_rows.len(), 2);
        assert_eq!(parsed_rows[0][0], "AAPL");
        assert_eq!(parsed_rows[1][2], "Contains, comma and \"quotes\"");
    }

    #[test]
    fn test_tsv_export() {
        let headers = vec!["ticker", "return"];
        let rows = vec![vec!["MSFT".to_string(), "0.012".to_string()]];
        let tsv = export_to_tsv(&headers, &rows);
        assert_eq!(tsv, "ticker\treturn\nMSFT\t0.012\n");
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_safetensors_roundtrip() {
        let tensor_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensors = vec![("layer1.weight", &[2, 2][..], &tensor_data[..])];

        let bytes = export_tensors_to_safetensors(&tensors).expect("Serialized safetensors");
        assert!(!bytes.is_empty());

        let loaded = import_tensors_from_safetensors(&bytes).expect("Loaded safetensors");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "layer1.weight");
        assert_eq!(loaded[0].1, vec![2, 2]);
        assert_eq!(loaded[0].2, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_wav_export() {
        let samples = vec![0.0f32, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let wav_bytes = export_to_wav_pcm32f(&samples, 44100, 1).expect("WAV created");
        assert!(wav_bytes.len() > 44);
        assert_eq!(&wav_bytes[0..4], b"RIFF");
        assert_eq!(&wav_bytes[8..12], b"WAVE");
        assert_eq!(&wav_bytes[12..16], b"fmt ");
    }

    #[test]
    fn test_wav_pcm16i_and_borrowed_csv() {
        let samples = vec![0.0f32, 0.5, 1.0, -1.0];
        let wav16 = export_to_wav_pcm16i(&samples, 48000, 1).expect("WAV 16i created");
        assert_eq!(&wav16[0..4], b"RIFF");
        assert_eq!(&wav16[8..12], b"WAVE");
        // AudioFormat should be 1 (PCM)
        let fmt_code = u16::from_le_bytes([wav16[20], wav16[21]]);
        assert_eq!(fmt_code, 1);
        let bits_per_sample = u16::from_le_bytes([wav16[34], wav16[35]]);
        assert_eq!(bits_per_sample, 16);

        // Zero-copy CSV line
        let line = "BHP.AX, 42.50, 10000";
        let fields = parse_csv_line_borrowed(line);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "BHP.AX");
        assert_eq!(fields[1], "42.50");
        assert!(matches!(fields[0], std::borrow::Cow::Borrowed(_)));
    }
}
