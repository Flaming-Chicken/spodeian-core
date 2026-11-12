//! Deterministic Preprocessing, Kahan Compensated Reductions & Hash Partitioning in Candle.
//!
//! Topic 08: 08_reproducible_preprocessing_pipelines_and_determinism (Category 00: Foundations)
//!
//! # Theoretical Overview
//! 1. Floating-Point Non-Associativity:
//!    Standard floating-point summation loses lower-order bits due to rounding errors:
//!    $$(a + b) + c \ne a + (b + c)$$
//!    Kahan summation preserves numerical precision by accumulating lost lower-order bits in a
//!    running compensation scalar $c$, bounding error to $\mathcal{O}(\epsilon_{\text{mach}} + N \epsilon_{\text{mach}}^2)$.
//! 2. Seed-State Encapsulation:
//!    Ensures reproducible forward inference without reliance on hidden global PRNG registers.
//! 3. Cryptographic/Deterministic Dataset Partitioning:
//!    Partitions samples using an FNV-1a salted hash digest, ensuring invariant assignment
//!    independent of dataset order or batching.
//!
//! # Examples
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t08_reproducible_preprocessing_pipelines_and_determinism::{
//!     DeterministicLinear, hash_deterministic_split, kahan_sum_2d
//! };
//!
//! # fn main() -> candle_core::Result<()> {
//! let dev = Device::Cpu;
//! let w = Tensor::zeros((4, 2), candle_core::DType::F32, &dev)?;
//! let b = Tensor::zeros(2, candle_core::DType::F32, &dev)?;
//! let layer = DeterministicLinear::new(w, b);
//!
//! let x = Tensor::zeros((8, 4), candle_core::DType::F32, &dev)?;
//! let y = layer.forward(&x)?;
//! assert_eq!(y.dims(), &[8, 2]);
//!
//! let split = hash_deterministic_split("row_42", "seed_salt", 0.8);
//! assert!(split == "train" || split == "test");
//! # Ok(())
//! # }
//! ```

use candle_core::{Result, Tensor};

/// Fast, architecture-independent 64-bit FNV-1a hash function.
pub fn fnv1a_hash(data: &str, salt: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in salt
        .bytes()
        .chain(std::iter::once(b':'))
        .chain(data.bytes())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Deterministically partitions a sample key into "train" or "test" using FNV-1a hashing.
pub fn hash_deterministic_split(sample_id: &str, salt: &str, train_ratio: f64) -> &'static str {
    let h = fnv1a_hash(sample_id, salt);
    let fraction = (h as f64) / (u64::MAX as f64);
    if fraction < train_ratio {
        "train"
    } else {
        "test"
    }
}

/// Evaluates Kahan compensated summation along dimension 0 of a 2D tensor [batch, dim].
pub fn kahan_sum_2d(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() != 2 {
        candle_core::bail!("kahan_sum_2d expects 2D tensor, got {:?}", dims);
    }
    let rows = dims[0];
    let cols = dims[1];
    let data = x.to_vec2::<f32>()?;

    let mut sum = vec![0.0f32; cols];
    let mut c = vec![0.0f32; cols];

    for i in 0..rows {
        for j in 0..cols {
            let y = data[i][j] - c[j];
            let t = sum[j] + y;
            c[j] = (t - sum[j]) - y;
            sum[j] = t;
        }
    }

    Tensor::from_vec(sum, cols, x.device())
}

/// Deterministic linear module.
pub struct DeterministicLinear {
    pub weight: Tensor, // [in_dim, out_dim]
    pub bias: Tensor,   // [out_dim]
}

impl DeterministicLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    /// Evaluates deterministic affine projection: y = x @ W + b.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_kahan_summation_precision() -> Result<()> {
        let dev = Device::Cpu;
        let data = vec![
            vec![1.0f32, 1e7, -1e-4],
            vec![1e-7f32, 1.0, 1e-4],
            vec![1e-7f32, -1e7, 1.0],
        ];
        let x = Tensor::new(data, &dev)?;
        let k_sum = kahan_sum_2d(&x)?;
        let k_vec = k_sum.to_vec1::<f32>()?;

        assert!((k_vec[0] - (1.0 + 2e-7)).abs() < 1e-6);
        assert!((k_vec[1] - 1.0).abs() < 1e-5);
        assert!((k_vec[2] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_fnv1a_determinism() {
        let h1 = fnv1a_hash("user_1024", "experiment_alpha");
        let h2 = fnv1a_hash("user_1024", "experiment_alpha");
        assert_eq!(h1, h2);

        let split1 = hash_deterministic_split("user_1024", "experiment_alpha", 0.8);
        let split2 = hash_deterministic_split("user_1024", "experiment_alpha", 0.8);
        assert_eq!(split1, split2);
    }
}
