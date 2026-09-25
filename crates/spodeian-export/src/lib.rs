//! Universal serialization and interchange utilities for Spodeian projects.
//!
//! Provides CSV field escaping, RFC 4180-compliant quote parsing, JSON formatted exports,
//! and high-ratio Zlib-compressed BSON binary exports.

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExportError {
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("BSON serialization error: {0}")]
    BsonSer(#[from] bson::ser::Error),
    #[error("BSON deserialization error: {0}")]
    BsonDe(#[from] bson::de::Error),
    #[error("CSV parse error: {0}")]
    Csv(String),
}

/// Escapes a string field for CSV inclusion following standard RFC 4180 rules.
pub fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}

/// Parses a single CSV line into discrete fields respecting quoted delimiters and escaped quotes.
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

/// Exports any serializable data structure into formatted JSON string.
pub fn export_to_json<T: Serialize>(data: &T) -> Result<String, ExportError> {
    serde_json::to_string_pretty(data).map_err(ExportError::Json)
}

/// Imports any deserializable data structure from a JSON string.
pub fn import_from_json<T: DeserializeOwned>(json_str: &str) -> Result<T, ExportError> {
    serde_json::from_str(json_str).map_err(ExportError::Json)
}

/// Exports any serializable data structure into compressed BSON binary bytes (Zlib-compressed BSON).
pub fn export_to_compressed_bson<T: Serialize>(data: &T) -> Result<Vec<u8>, ExportError> {
    let bson_bytes = bson::to_vec(data).map_err(ExportError::BsonSer)?;
    Ok(miniz_oxide::deflate::compress_to_vec_zlib(&bson_bytes, 6))
}

/// Imports and restores any deserializable data structure from a compressed (or raw) BSON slice.
pub fn import_from_compressed_bson<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ExportError> {
    let bson_bytes =
        miniz_oxide::inflate::decompress_to_vec_zlib(bytes).unwrap_or_else(|_| bytes.to_vec());
    bson::from_slice::<T>(&bson_bytes).map_err(ExportError::BsonDe)
}
