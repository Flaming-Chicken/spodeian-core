//! GGUF v2/v3 Serialization and Streaming Loader (`sml-runtime::format::gguf`).
//!
//! Provides zero-copy reading, metadata extraction, and binary writing for standard
//! GGUF files compatible with KoboldCpp, llama.cpp, Ollama, and LM Studio.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use thiserror::Error;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("Invalid GGUF magic: expected 0x46554747, got 0x{0:08x}")]
    InvalidMagic(u32),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Format error: {0}")]
    Format(String),
}

/// GGUF Value Types according to GGUF Specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    pub fn from_u32(val: u32) -> Result<Self, GgufError> {
        match val {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            other => Err(GgufError::Format(format!("Unknown GGUF type ID: {other}"))),
        }
    }
}

/// Metadata key-value storage.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor descriptor in GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// Parsed GGUF Model File Content.
#[derive(Debug, Clone)]
pub struct GgufModel {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub alignment: usize,
}

impl GgufModel {
    /// Reads and parses GGUF header and metadata from a reader.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(reader)?;
            let type_id = reader.read_u32::<LittleEndian>()?;
            let val_type = GgufType::from_u32(type_id)?;
            let value = read_gguf_value(reader, val_type)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                GgufValue::Uint32(a) => Some(*a as usize),
                GgufValue::Uint64(a) => Some(*a as usize),
                _ => None,
            })
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()? as usize);
            }
            let ggml_type = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    ggml_type,
                    offset,
                },
            );
        }

        Ok(Self {
            version,
            metadata,
            tensors,
            alignment,
        })
    }

    /// Queries the model's architecture string (e.g. "llama", "qwen2", "dit").
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Queries the model's context length.
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture().unwrap_or("llama");
        let key = format!("{arch}.context_length");
        match self.metadata.get(&key) {
            Some(GgufValue::Uint32(v)) => Some(*v as usize),
            Some(GgufValue::Uint64(v)) => Some(*v as usize),
            _ => None,
        }
    }
}

/// Helper function to write a string in GGUF format: [length: u64][bytes].
pub fn write_gguf_string<W: Write>(writer: &mut W, s: &str) -> Result<(), std::io::Error> {
    writer.write_u64::<LittleEndian>(s.len() as u64)?;
    writer.write_all(s.as_bytes())?;
    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| GgufError::Format(format!("Invalid UTF-8 string: {e}")))
}

fn read_gguf_value<R: Read>(reader: &mut R, val_type: GgufType) -> Result<GgufValue, GgufError> {
    match val_type {
        GgufType::Uint8 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        GgufType::Int8 => Ok(GgufValue::Int8(reader.read_i8()?)),
        GgufType::Uint16 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufType::Int16 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufType::Uint32 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufType::Int32 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufType::Float32 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufType::Bool => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        GgufType::String => Ok(GgufValue::String(read_gguf_string(reader)?)),
        GgufType::Uint64 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufType::Int64 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufType::Float64 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
        GgufType::Array => {
            let item_type_id = reader.read_u32::<LittleEndian>()?;
            let item_type = GgufType::from_u32(item_type_id)?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_gguf_value(reader, item_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
    }
}
