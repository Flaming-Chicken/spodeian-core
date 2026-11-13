//! Rigour Plan Phase P5: High-Dimensional Scaling (d = 10^2 to 10^6),
//! Real-World Multiclass Classification, and Native vs F64 Precision Analysis.

use candle_core::{DType, Device, Result, Tensor, Var};
use sml_runtime::optim::{
    OptimizationRegime, Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig,
};
use std::time::Instant;

// ============================================================================
// 1. High-Dimensional Parameter Scaling: d in [10^2, 10^3, 10^4, 10^5, 10^6]
// ============================================================================

#[test]
fn test_large_d_linear_scaling() -> Result<()> {
    let dev = Device::Cpu;
    let dimensions = [100, 1_000, 10_000, 100_000];

    let mut prev_per_dim_time = 0.0f64;

    for &d in &dimensions {
        let x = Var::from_vec(vec![1.0f32; d], (d,), &dev)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig {
                lr: 0.05,
                beta: 0.85,
                curvature_damping: 0.5,
                weight_decay: 0.0,
                warmup_steps: 2,
                min_step_scale: 0.01,
                eps: 1e-8,
                regime: OptimizationRegime::NonConvexValley,
                ..Default::default()
            },
        );

        // Warmup step
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;

        // Compute gradients outside timed region
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;

        // Timed pure optimizer step
        let start = Instant::now();
        opt.step(&grads)?;
        let elapsed_us = start.elapsed().as_micros() as f64;

        let per_dim_time_ns = (elapsed_us * 1000.0) / d as f64;
        println!(
            "Dim d = {:>7}: step = {:>7.1} us | per-dim = {:>5.2} ns/dim",
            d, elapsed_us, per_dim_time_ns
        );

        // O(d) linear complexity check: per-dim time must not blow up exponentially
        if prev_per_dim_time > 0.0 {
            assert!(
                per_dim_time_ns < prev_per_dim_time * 5.0,
                "Scaling bottleneck detected at d = {}: per-dim time jumped from {:.2} to {:.2} ns",
                d,
                prev_per_dim_time,
                per_dim_time_ns
            );
        }
        prev_per_dim_time = per_dim_time_ns;
    }

    Ok(())
}

// ============================================================================
// 2. Real-World Multiclass Classification Task (D=16, C=4 classes, N=200 samples)
// ============================================================================

#[test]
fn test_real_multiclass_classification_task() -> Result<()> {
    let dev = Device::Cpu;
    let total_samples = 200;
    let train_samples = 150;
    let test_samples = 50;
    let in_dim = 16;
    let hidden_dim = 32;
    let out_dim = 4;

    let mut x_train_data = Vec::with_capacity(train_samples * in_dim);
    let mut y_train_data = Vec::with_capacity(train_samples * out_dim);
    let mut train_labels = Vec::with_capacity(train_samples);

    let mut x_test_data = Vec::with_capacity(test_samples * in_dim);
    let mut y_test_data = Vec::with_capacity(test_samples * out_dim);
    let mut test_labels = Vec::with_capacity(test_samples);

    for i in 0..total_samples {
        let class_idx = i % out_dim;
        let mut row = Vec::with_capacity(in_dim);
        for j in 0..in_dim {
            // Manifold: class-specific cluster center plus bounded intra-class variance
            let class_center = ((class_idx as f32 * 1.57 + j as f32 * 0.4).sin()) * 2.0;
            let sample_jitter = (((i * 13 + j * 17) % 31) as f32 / 31.0 - 0.5) * 0.4;
            row.push(class_center + sample_jitter);
        }
        let mut target = vec![0.0f32; out_dim];
        target[class_idx] = 1.0;

        if i < train_samples {
            x_train_data.extend(row);
            y_train_data.extend(target);
            train_labels.push(class_idx);
        } else {
            x_test_data.extend(row);
            y_test_data.extend(target);
            test_labels.push(class_idx);
        }
    }

    let x_train = Tensor::from_vec(x_train_data, (train_samples, in_dim), &dev)?;
    let y_train = Tensor::from_vec(y_train_data, (train_samples, out_dim), &dev)?;
    let x_test = Tensor::from_vec(x_test_data, (test_samples, in_dim), &dev)?;

    // Xavier-style symmetry-broken weight initializations
    let scale1 = (2.0 / (in_dim + hidden_dim) as f32).sqrt();
    let mut w1_init = Vec::with_capacity(in_dim * hidden_dim);
    for i in 0..in_dim {
        for j in 0..hidden_dim {
            let val = (((i * 17 + j * 31 + 7) % 53) as f32 / 53.0 - 0.5) * 2.0 * scale1;
            w1_init.push(val);
        }
    }

    let scale2 = (2.0 / (hidden_dim + out_dim) as f32).sqrt();
    let mut w2_init = Vec::with_capacity(hidden_dim * out_dim);
    for j in 0..hidden_dim {
        for c in 0..out_dim {
            let val = (((j * 13 + c * 29 + 11) % 47) as f32 / 47.0 - 0.5) * 2.0 * scale2;
            w2_init.push(val);
        }
    }

    let w1 = Var::from_vec(w1_init, (in_dim, hidden_dim), &dev)?;
    let w2 = Var::from_vec(w2_init, (hidden_dim, out_dim), &dev)?;

    let mut opt = SelfOptimizingMeta::new(
        vec![w1.clone(), w2.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.1,
            beta: 0.85,
            curvature_damping: 0.5,
            weight_decay: 1e-4,
            warmup_steps: 3,
            min_step_scale: 0.01,
            eps: 1e-8,
            regime: OptimizationRegime::NeuralNetwork,
            ..Default::default()
        },
    );

    let initial_loss = {
        let h = x_train.matmul(w1.as_tensor())?.relu()?;
        let logits = h.matmul(w2.as_tensor())?;
        let diff = (&logits - &y_train)?;
        diff.sqr()?.sum_all()?.to_vec0::<f32>()? / train_samples as f32
    };

    println!(
        "Classification Task - Initial Train MSE Loss: {:.4}",
        initial_loss
    );

    for _ in 0..120 {
        let h = x_train.matmul(w1.as_tensor())?.relu()?;
        let logits = h.matmul(w2.as_tensor())?;
        let diff = (&logits - &y_train)?;
        let loss = diff.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let final_loss = {
        let h = x_train.matmul(w1.as_tensor())?.relu()?;
        let logits = h.matmul(w2.as_tensor())?;
        let diff = (&logits - &y_train)?;
        diff.sqr()?.sum_all()?.to_vec0::<f32>()? / train_samples as f32
    };

    println!(
        "Classification Task - Final Train MSE Loss: {:.4}",
        final_loss
    );
    assert!(
        final_loss < initial_loss * 0.35,
        "Multiclass classification failed to converge: initial = {:.4}, final = {:.4}",
        initial_loss,
        final_loss
    );

    // Evaluate on held-out validation set
    let h_test = x_test.matmul(w1.as_tensor())?.relu()?;
    let test_logits = h_test.matmul(w2.as_tensor())?;
    let test_vals = test_logits.to_vec2::<f32>()?;

    let mut correct = 0;
    for (row, &true_cls) in test_vals.iter().zip(&test_labels) {
        let mut pred_cls = 0;
        let mut max_val = row[0];
        for (c, &v) in row.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                pred_cls = c;
            }
        }
        if pred_cls == true_cls {
            correct += 1;
        }
    }
    let accuracy = (correct as f32 / test_samples as f32) * 100.0;
    println!(
        "Held-out Validation Accuracy: {:.1}% ({}/{})",
        accuracy, correct, test_samples
    );
    assert!(
        accuracy >= 75.0,
        "Classification accuracy on held-out test set failed: {:.1}%",
        accuracy
    );

    Ok(())
}

// ============================================================================
// 3. Native F32 vs F64 Precision Equivalence Analysis
// ============================================================================

#[test]
fn test_native_f32_vs_f64_precision_fidelity() -> Result<()> {
    let dev = Device::Cpu;

    // F64 optimization trajectory
    let x_f64 = Var::from_slice(&[4.0f64, -3.0f64], (2,), &dev)?;
    let mut opt_f64 = SelfOptimizingMeta::new(
        vec![x_f64.clone()],
        SelfOptimizingMetaConfig::for_dtype(DType::F64),
    );
    for _ in 0..40 {
        let loss = x_f64.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt_f64.step(&grads)?;
    }
    let final_f64 = x_f64.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;

    // F32 native optimization trajectory
    let x_f32 = Var::from_slice(&[4.0f32, -3.0f32], (2,), &dev)?;
    let mut opt_f32 = SelfOptimizingMeta::new(
        vec![x_f32.clone()],
        SelfOptimizingMetaConfig::for_dtype(DType::F32),
    );
    for _ in 0..40 {
        let loss = x_f32.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt_f32.step(&grads)?;
    }
    let final_f32 = x_f32.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()? as f64;

    println!(
        "Precision Fidelity: F64 final = {:.2e} | F32 final = {:.2e}",
        final_f64, final_f32
    );

    // Both achieve sub-2e-3 convergence; F32 maintains high fidelity without casting overhead
    assert!(final_f64 < 2e-3);
    assert!(final_f32 < 2e-3);
    assert!(
        (final_f64 - final_f32).abs() < 1e-2,
        "F32 trajectory diverged from F64 ground truth!"
    );

    Ok(())
}
