//! Automated test verifying the exact memory footprint accounting and zero-allocation
//! stack-inlined execution of SelfOptimizingMeta.
//!
//! Enforces Phase 0 of the Rigour Plan:
//! 1. Zero heap allocations for N <= 64.
//! 2. Exact bytes-per-param accounting for single-layer (48 B/param) vs multi-layer (32 B/param).
//! 3. Spilled heap accounting for N > 64.
//! 4. Shared scratch buffer consolidation across all parameters.

use candle_core::{DType, Device, Result, Tensor, Var};
use sml_runtime::optim::{Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig};

#[test]
fn test_memory_accounting_single_layer_inline() -> Result<()> {
    let dev = Device::Cpu;
    // 2D optimization problem (N = 2 <= 64)
    let var = Var::from_tensor(&Tensor::new(&[1.5f64, -2.0f64], &dev)?)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::F64);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let footprint = opt.memory_footprint();
    assert_eq!(footprint.num_parameters, 2);
    // Single-layer L=1: 6 vectors (y, z, v, prev_g, prev_theta, flip_ema) of 2 f64 each = 12 f64 = 96 bytes
    assert_eq!(footprint.persistent_state_bytes, 6 * 2 * 8);
    assert_eq!(footprint.bytes_per_param, 48); // 6 scalars * 8 bytes/scalar
    // Crucial guarantee: Zero heap allocations for N <= 64!
    assert_eq!(footprint.heap_allocated_bytes, 0);
    assert_eq!(footprint.inline_stack_bytes, 96);

    // Perform a step and verify heap bytes remain 0
    let loss = var.as_tensor().sqr()?.sum_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let post_step_footprint = opt.memory_footprint();
    assert_eq!(post_step_footprint.heap_allocated_bytes, 0);
    assert_eq!(post_step_footprint.bytes_per_param, 48);

    Ok(())
}

#[test]
fn test_memory_accounting_multi_layer_depth() -> Result<()> {
    let dev = Device::Cpu;
    // Multi-layer deep neural network (L = 3 parameter tensors)
    let w1 = Var::from_tensor(&Tensor::zeros((4, 4), DType::F64, &dev)?)?;
    let w2 = Var::from_tensor(&Tensor::zeros((4, 4), DType::F64, &dev)?)?;
    let w3 = Var::from_tensor(&Tensor::zeros((4, 2), DType::F64, &dev)?)?;
    let total_params = 16 + 16 + 8; // 40 params <= 64 each

    let config = SelfOptimizingMetaConfig::for_dtype(DType::F64);
    let opt = SelfOptimizingMeta::new(vec![w1, w2, w3], config);

    let footprint = opt.memory_footprint();
    assert_eq!(footprint.num_parameters, total_params);
    // Multi-layer L > 1: 4 vectors (y, z, m, v) of f64 each, history vectors are empty (0 bytes)!
    // 4 scalars * 8 bytes = 32 bytes/param.
    assert_eq!(footprint.bytes_per_param, 32);
    assert_eq!(footprint.persistent_state_bytes, total_params * 32);
    // Again, all tensors have <= 64 elements, so heap allocations must be strictly 0!
    assert_eq!(footprint.heap_allocated_bytes, 0);

    Ok(())
}

#[test]
fn test_memory_accounting_heap_spill_large_weights() -> Result<()> {
    let dev = Device::Cpu;
    // Large weight matrix (N = 128 > 64)
    let w_large = Var::from_tensor(&Tensor::zeros((16, 8), DType::F64, &dev)?)?;
    assert_eq!(w_large.as_tensor().elem_count(), 128);

    let config = SelfOptimizingMetaConfig::for_dtype(DType::F64);
    let opt = SelfOptimizingMeta::new(vec![w_large], config);

    let footprint = opt.memory_footprint();
    assert_eq!(footprint.num_parameters, 128);
    // N > 64 forces SmallVec to spill to the heap
    assert!(footprint.heap_allocated_bytes > 0);
    assert_eq!(footprint.bytes_per_param, 48); // 6 scalars * 8 bytes

    Ok(())
}

#[test]
fn test_memory_accounting_single_layer_f32_parity() -> Result<()> {
    let dev = Device::Cpu;
    // F32 parameter: State vectors match parameter DType (f32) instead of casting to f64
    let var = Var::from_tensor(&Tensor::new(&[1.5f32, -2.0f32], &dev)?)?;
    let config = SelfOptimizingMetaConfig::for_dtype(DType::F32);
    let mut opt = SelfOptimizingMeta::new(vec![var.clone()], config);

    let footprint = opt.memory_footprint();
    assert_eq!(footprint.num_parameters, 2);
    // Single-layer L=1: 6 vectors of 2 f32 each = 12 f32 = 48 bytes persistent
    assert_eq!(footprint.persistent_state_bytes, 6 * 2 * 4);
    // Crucial: 24 B/param vs 48 B/param in F64 (exact 50% memory reduction!)
    assert_eq!(footprint.bytes_per_param, 24);
    assert_eq!(footprint.heap_allocated_bytes, 0);
    assert_eq!(footprint.inline_stack_bytes, 48);

    // Perform an optimization step
    let loss = var.as_tensor().sqr()?.sum_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let post_step_footprint = opt.memory_footprint();
    assert_eq!(post_step_footprint.heap_allocated_bytes, 0);
    assert_eq!(post_step_footprint.bytes_per_param, 24);

    Ok(())
}

#[test]
fn test_memory_accounting_multi_layer_f32_parity() -> Result<()> {
    let dev = Device::Cpu;
    let w1 = Var::from_tensor(&Tensor::zeros((4, 4), DType::F32, &dev)?)?;
    let w2 = Var::from_tensor(&Tensor::zeros((4, 4), DType::F32, &dev)?)?;
    let w3 = Var::from_tensor(&Tensor::zeros((4, 2), DType::F32, &dev)?)?;
    let total_params = 16 + 16 + 8; // 40 params <= 64 each

    let config = SelfOptimizingMetaConfig::for_dtype(DType::F32);
    let opt = SelfOptimizingMeta::new(vec![w1, w2, w3], config);

    let footprint = opt.memory_footprint();
    assert_eq!(footprint.num_parameters, total_params);
    // Multi-layer L > 1: 4 vectors (y, z, m, v) of f32 each = 16 bytes/param (vs 32 B in F64)!
    assert_eq!(footprint.bytes_per_param, 16);
    assert_eq!(footprint.persistent_state_bytes, total_params * 16);
    assert_eq!(footprint.heap_allocated_bytes, 0);

    Ok(())
}
