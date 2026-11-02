//! Topic 02: Tensors, Views, Strides & Memory Layouts (Native Rust Candle Track).
//!
//! Demonstrates physical tensor memory layouts, linear stride arithmetic,
//! contiguous vs transposed non-contiguous tensors, and sliding window extractions.

use candle_core::{Result, Tensor};

/// Utility for analyzing memory layout and stride arithmetic.
///
/// # Example
/// ```rust
/// use ai_candle::c00_foundations::t02_tensors_views_strides_and_memory_layouts::MemoryLayout;
///
/// let shape = vec![2, 3, 4];
/// let strides = MemoryLayout::standard_strides(&shape);
/// assert_eq!(strides, vec![12, 4, 1]);
/// let idx = MemoryLayout::linear_index(&[1, 2, 3], &strides);
/// assert_eq!(idx, 1 * 12 + 2 * 4 + 3 * 1);
/// ```
pub struct MemoryLayout;

impl MemoryLayout {
    /// Computes the 1D physical element index given coordinates and strides:
    /// index = sum_k (coords[k] * strides[k])
    pub fn linear_index(coords: &[usize], strides: &[usize]) -> usize {
        coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum()
    }

    /// Computes standard Row-Major (C-contiguous) strides for a given shape.
    pub fn standard_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return Vec::new();
        }
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Extracts sliding window chunks from a 1D tensor along the last dimension.
    pub fn sliding_windows_1d(x: &Tensor, window_size: usize, step: usize) -> Result<Tensor> {
        let (_b, len) = x.dims2()?;
        let num_windows = (len - window_size) / step + 1;
        let mut windows = Vec::with_capacity(num_windows);

        for w in 0..num_windows {
            let start = w * step;
            let slice = x.narrow(1, start, window_size)?;
            windows.push(slice);
        }
        // Stack windows along dimension 1 -> [B, Num_Windows, Window_Size]
        Tensor::stack(&windows, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_stride_arithmetic() {
        let shape = [2, 3, 4];
        let strides = MemoryLayout::standard_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);

        let coord = [1, 2, 3];
        let idx = MemoryLayout::linear_index(&coord, &strides);
        assert_eq!(idx, 23);
    }

    #[test]
    fn test_sliding_windows_shape() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, 16f32, &device)?.reshape((1, 16))?;
        let windows = MemoryLayout::sliding_windows_1d(&x, 4, 2)?;

        assert_eq!(windows.dims(), &[1, 7, 4]);
        Ok(())
    }
}
