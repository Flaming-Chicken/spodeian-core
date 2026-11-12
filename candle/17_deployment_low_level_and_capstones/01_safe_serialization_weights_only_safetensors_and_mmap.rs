//! Safe Serialization Weights Only Safetensors and Mmap in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Evaluates linear projection using safely loaded model parameters.
pub fn project_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Extracts the 8-byte little-endian header length from safetensors raw bytes.
pub fn parse_safetensors_header_len(bytes: &[u8]) -> usize {
    if bytes.len() < 8 {
        return 0;
    }
    u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8])) as usize
}
