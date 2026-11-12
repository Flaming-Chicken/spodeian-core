//! Module Internals, Cayley Orthogonal Parametrizations & Meta Devices in Candle.
//!
//! Topic 06: 06_nn_module_internals_parametrizations_and_meta_device (Category 00: Foundations)
//!
//! # Theoretical Overview
//! 1. Manifold Parametrization:
//!    To guarantee strictly isometric linear transformations (preserving $\ell_2$ norms),
//!    weights are constrained to the Lie group $SO(n)$ using the Cayley transform:
//!    $$W = (I - A)(I + A)^{-1}$$
//!    where $A \in \mathfrak{so}(n)$ is a skew-symmetric matrix ($A = -A^T$) generated from
//!    the strictly upper triangular elements of an unconstrained parameter matrix $M$.
//! 2. Meta Device / Zero-Memory Tracing:
//!    Simulates tensor rank, shape propagation, and memory consumption across deep network
//!    topologies without allocating physical memory buffers.
//!
//! # Examples
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t06_nn_module_internals_parametrizations_and_meta_device::{
//!     cayley_transform, OrthogonalLinear, MetaTensor
//! };
//!
//! # fn main() -> candle_core::Result<()> {
//! let dev = Device::Cpu;
//! let m = Tensor::zeros((3, 3), candle_core::DType::F32, &dev)?;
//! let w = cayley_transform(&m)?;
//! assert_eq!(w.dims(), &[3, 3]);
//!
//! let meta = MetaTensor::new(&[8, 1024], candle_core::DType::F32);
//! let meta_out = meta.linear_forward(2048)?;
//! assert_eq!(meta_out.dims, vec![8, 2048]);
//! # Ok(())
//! # }
//! ```

use candle_core::{DType, Result, Tensor};

/// Inverts a square matrix using partial-pivoting Gauss-Jordan elimination.
pub fn invert_matrix(p: &Tensor) -> Result<Tensor> {
    let dims = p.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        candle_core::bail!("Matrix must be square, got {:?}", dims);
    }
    let n = dims[0];
    let data = p.to_vec2::<f32>()?;
    let mut aug = vec![vec![0.0f32; 2 * n]; n];

    for i in 0..n {
        for j in 0..n {
            aug[i][j] = data[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[pivot_row][i].abs() {
                pivot_row = k;
            }
        }
        if pivot_row != i {
            aug.swap(i, pivot_row);
        }
        let pivot = aug[i][i];
        if pivot.abs() < 1e-12 {
            candle_core::bail!("Singular matrix encountered during Gauss-Jordan inversion");
        }
        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = aug[r][i];
                for c in 0..(2 * n) {
                    aug[r][c] -= factor * aug[i][c];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Tensor::new(inv, p.device())
}

/// Evaluates Cayley orthogonal transform: W = (I - A)(I + A)^{-1}.
pub fn cayley_transform(m: &Tensor) -> Result<Tensor> {
    let dims = m.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        candle_core::bail!("Cayley transform requires square matrix, got {:?}", dims);
    }
    let n = dims[0];
    let m_data = m.to_vec2::<f32>()?;

    let mut p_data = vec![vec![0.0f32; n]; n];
    let mut q_data = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        p_data[i][i] = 1.0;
        q_data[i][i] = 1.0;
        for j in (i + 1)..n {
            let val = m_data[i][j];
            p_data[i][j] = val;
            p_data[j][i] = -val;
            q_data[i][j] = -val;
            q_data[j][i] = val;
        }
    }

    let dev = m.device();
    let p = Tensor::new(p_data, dev)?;
    let q = Tensor::new(q_data, dev)?;
    let p_inv = invert_matrix(&p)?;

    q.matmul(&p_inv)
}

/// Linear layer constrained to orthogonal parameter manifold.
pub struct OrthogonalLinear {
    pub skew_param: Tensor, // [dim, dim]
    pub bias: Tensor,       // [dim]
}

impl OrthogonalLinear {
    pub fn new(skew_param: Tensor, bias: Tensor) -> Self {
        Self { skew_param, bias }
    }

    /// Forward affine projection with orthogonal weight matrix: y = x @ W + b.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = cayley_transform(&self.skew_param)?;
        x.matmul(&w)?.broadcast_add(&self.bias)
    }

    /// Returns the effective computed orthogonal weight tensor W.
    pub fn effective_weight(&self) -> Result<Tensor> {
        cayley_transform(&self.skew_param)
    }
}

/// Abstract shape tensor simulating zero-allocation 'meta' device propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaTensor {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl MetaTensor {
    pub fn new(dims: &[usize], dtype: DType) -> Self {
        Self {
            dims: dims.to_vec(),
            dtype,
        }
    }

    /// Simulates linear projection metadata without physical memory allocation.
    pub fn linear_forward(&self, out_features: usize) -> Result<Self> {
        if self.dims.is_empty() {
            candle_core::bail!("MetaTensor must have at least 1 dimension");
        }
        let mut new_dims = self.dims.clone();
        let last_idx = new_dims.len() - 1;
        new_dims[last_idx] = out_features;
        Ok(Self {
            dims: new_dims,
            dtype: self.dtype,
        })
    }

    /// Propagates shapes through an arbitrary number of linear layers.
    pub fn propagate_deep_stack(&self, hidden_dim: usize, num_layers: usize) -> Result<Self> {
        let mut current = self.clone();
        for _ in 0..num_layers {
            current = current.linear_forward(hidden_dim)?;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cayley_orthogonality() -> Result<()> {
        let dev = Device::Cpu;
        let m_data = vec![
            vec![0.5f32, 1.2, -0.4],
            vec![0.1, -0.8, 0.7],
            vec![0.3, 0.9, -0.2],
        ];
        let m = Tensor::new(m_data, &dev)?;
        let w = cayley_transform(&m)?;

        // W^T @ W == I
        let wt = w.t()?;
        let id_check = wt.matmul(&w)?;
        let eye = Tensor::eye(3, DType::F32, &dev)?;
        let max_err = id_check
            .sub(&eye)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_err < 1e-6,
            "Orthogonality error {:.2e} exceeds 1e-6",
            max_err
        );
        Ok(())
    }

    #[test]
    fn test_meta_tensor_stack() -> Result<()> {
        let meta = MetaTensor::new(&[4, 512], DType::F32);
        let deep = meta.propagate_deep_stack(1024, 100)?;
        assert_eq!(deep.dims, vec![4, 1024]);
        assert_eq!(deep.dtype, DType::F32);
        Ok(())
    }
}
