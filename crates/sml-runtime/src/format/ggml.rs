//! Legacy GGML Binary Model Ingestion & Auto-Upgrade (`sml-runtime::format::ggml`).
//!
//! Detects and parses legacy GGML binary formats (`ggml`, `ggmf`, `ggjt` v1..v3)
//! used by older versions of llama.cpp, whisper.cpp, and early KoboldCpp,
//! allowing automatic conversion and upgrade to modern GGUF v3 and SafeTensors.

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek};
use thiserror::Error;

pub const GGML_MAGIC_GGML: u32 = 0x67676d6c; // "ggml"
pub const GGML_MAGIC_GGMF: u32 = 0x67676d66; // "ggmf"
pub const GGML_MAGIC_GGJT: u32 = 0x67676a74; // "ggjt"

#[derive(Error, Debug)]
pub enum GgmlError {
    #[error("Not a recognized GGML file (magic 0x{0:08x})")]
    NotGgml(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Format error: {0}")]
    Format(String),
}

/// Legacy GGML Model Header.
#[derive(Debug, Clone)]
pub struct GgmlHeader {
    pub magic: u32,
    pub version: u32,
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

/// Legacy GGML Model Descriptor.
#[derive(Debug, Clone)]
pub struct GgmlModel {
    pub header: GgmlHeader,
    pub vocabulary: Vec<String>,
    pub tensor_names: Vec<String>,
}

impl GgmlModel {
    /// Detects whether the input stream begins with a recognized GGML magic number.
    pub fn is_ggml<R: Read + Seek>(reader: &mut R) -> bool {
        let pos = reader.stream_position().unwrap_or(0);
        let magic = reader.read_u32::<LittleEndian>().unwrap_or(0);
        let _ = reader.seek(std::io::SeekFrom::Start(pos));
        matches!(magic, GGML_MAGIC_GGML | GGML_MAGIC_GGMF | GGML_MAGIC_GGJT)
    }

    /// Ingests and parses legacy GGML file structure.
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self, GgmlError> {
        let magic = reader.read_u32::<LittleEndian>()?;
        if !matches!(magic, GGML_MAGIC_GGML | GGML_MAGIC_GGMF | GGML_MAGIC_GGJT) {
            return Err(GgmlError::NotGgml(magic));
        }

        let version = if magic == GGML_MAGIC_GGJT || magic == GGML_MAGIC_GGMF {
            reader.read_u32::<LittleEndian>()?
        } else {
            1
        };

        let n_vocab = reader.read_u32::<LittleEndian>()?;
        let n_embd = reader.read_u32::<LittleEndian>()?;
        let n_mult = reader.read_u32::<LittleEndian>()?;
        let n_head = reader.read_u32::<LittleEndian>()?;
        let n_layer = reader.read_u32::<LittleEndian>()?;
        let n_rot = reader.read_u32::<LittleEndian>()?;
        let ftype = reader.read_u32::<LittleEndian>()?;

        let header = GgmlHeader {
            magic,
            version,
            n_vocab,
            n_embd,
            n_mult,
            n_head,
            n_layer,
            n_rot,
            ftype,
        };

        // Read vocabulary tokens
        let mut vocabulary = Vec::with_capacity(n_vocab as usize);
        for _ in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            let token = String::from_utf8_lossy(&buf).to_string();
            // In GGJT, each token also has a float score
            if magic == GGML_MAGIC_GGJT {
                let _score = reader.read_f32::<LittleEndian>()?;
            }
            vocabulary.push(token);
        }

        Ok(Self {
            header,
            vocabulary,
            tensor_names: Vec::new(),
        })
    }
}
