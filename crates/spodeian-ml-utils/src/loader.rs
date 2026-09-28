//! SafeTensors streaming loader and zero-copy weight ingestion for Candle.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MlUtilsError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Missing tensor '{0}'")]
    MissingTensor(String),
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Dequantization error: {0}")]
    Dequantization(String),
}

/// Container for loaded model weights with zero-copy access and metadata.
#[derive(Clone)]
pub struct LoadedModelWeights {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
    pub device: Device,
}

impl LoadedModelWeights {
    /// Ingest a SafeTensors binary buffer directly into Candle tensors.
    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, MlUtilsError> {
        let tensors = candle_core::safetensors::load_buffer(bytes, device)?;
        let metadata = HashMap::new();

        Ok(Self {
            tensors,
            metadata,
            device: device.clone(),
        })
    }

    /// Load weights from a local SafeTensors file.
    pub fn from_file<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self, MlUtilsError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes, device)
    }

    /// Check if a named tensor exists.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get a reference to a loaded tensor.
    pub fn get_tensor(&self, name: &str) -> Result<&Tensor, MlUtilsError> {
        self.tensors
            .get(name)
            .ok_or_else(|| MlUtilsError::MissingTensor(name.to_string()))
    }

    /// Create a Candle `VarBuilder` backed by these loaded tensors.
    pub fn var_builder(&self, dtype: DType) -> VarBuilder<'_> {
        VarBuilder::from_tensors(self.tensors.clone(), dtype, &self.device)
    }

    /// Total number of stored tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the weight container is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Estimate total memory footprint in bytes.
    pub fn memory_size_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|t| {
                let elem_count = t.shape().elem_count();
                let dtype_size = match t.dtype() {
                    DType::F32 | DType::U32 | DType::I64 => 4,
                    DType::F64 => 8,
                    DType::F16 | DType::BF16 => 2,
                    DType::U8 => 1,
                    _ => 4,
                };
                elem_count * dtype_size
            })
            .sum()
    }

    /// Dequantize an INT8 (or U8) tensor into F32 using a scale factor:
    /// $$w_{f32} = w_{int8} \times \text{scale}$$
    pub fn dequantize_int8(tensor: &Tensor, scale: f32) -> Result<Tensor, MlUtilsError> {
        let f32_tensor = tensor.to_dtype(DType::F32)?;
        let scaled = (f32_tensor * (scale as f64))?;
        Ok(scaled)
    }
}
