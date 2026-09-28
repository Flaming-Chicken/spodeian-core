//! Comprehensive tests for spodeian-ml-utils.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use spodeian_ml_utils::{
    DualSpeedAdaptationManager, DynamicMultiHorizonHead, LoadedModelWeights, MonotonicQuantileHead,
    OnlineRlsHead, SharpnessAwareMinimizer, VelocityFlowHead,
};
use std::collections::HashMap;

#[test]
fn test_online_rls_convergence() {
    let mut rls = OnlineRlsHead::new(3, 0.98, 1000.0);
    let true_weights = [2.5f32, -1.2f32, 0.8f32];

    let mut last_error = 10.0f32;
    for step in 0..200 {
        let x = [
            (step as f32 * 0.1).sin(),
            (step as f32 * 0.15).cos(),
            (step as f32 * 0.2).sin(),
        ];
        let true_y = x[0] * true_weights[0] + x[1] * true_weights[1] + x[2] * true_weights[2];

        last_error = rls.update(&x, true_y);
    }

    assert!(
        last_error.abs() < 0.05,
        "RLS error did not converge: {last_error}"
    );

    for (w_hat, &w_true) in rls.weights.iter().zip(&true_weights) {
        assert!(
            (w_hat - w_true).abs() < 0.1,
            "Weight estimate mismatch: got {w_hat}, expected {w_true}"
        );
    }
}

#[test]
fn test_sam_clipping_and_perturbation() {
    let sam = SharpnessAwareMinimizer::new(0.05, 1.0);
    let mut grad = vec![3.0f32, 4.0f32]; // Norm is 5.0

    let norm_before = sam.clip_gradient(&mut grad);
    assert!((norm_before - 5.0).abs() < 1e-4);

    let norm_after = SharpnessAwareMinimizer::grad_norm(&grad);
    assert!(
        (norm_after - 1.0).abs() < 1e-4,
        "Clipped gradient norm should be 1.0, got {norm_after}"
    );

    let pert = sam.compute_perturbation(&grad);
    let pert_norm = SharpnessAwareMinimizer::grad_norm(&pert);
    assert!(
        (pert_norm - 0.05).abs() < 1e-4,
        "Perturbation norm should equal rho (0.05), got {pert_norm}"
    );
}

#[test]
fn test_dual_speed_adaptation_anomaly_and_utility() {
    let mut manager = DualSpeedAdaptationManager::new(2, 0.001, 0.99, 0.05, 1.0);

    let x_normal = [1.0f32, 0.5f32];
    for _ in 0..50 {
        manager.step(&x_normal, 2.0);
    }

    // Step with a massive shock
    let (_pred, error, is_anomaly, utility) = manager.step(&x_normal, 50.0);
    assert!(error > 40.0);
    assert!(is_anomaly, "Shock should trigger anomaly flag");
    assert!(
        utility > 0.8,
        "Shock should yield high training utility, got {utility}"
    );

    // Test slow encoder step with SAM
    let mut weights = vec![1.0f32, -0.5f32];
    let mut grads = vec![10.0f32, 5.0f32];
    manager.slow_encoder_step(&mut weights, &mut grads);
    assert!(weights[0] < 1.0);
}

#[test]
fn test_dynamic_multi_horizon_head() {
    let device = Device::Cpu;
    let horizons = vec![1, 2, 5, 10, 21, 63, 252];
    let input_dim = 8;

    let mut tensors = HashMap::new();
    let weight = Tensor::zeros((horizons.len(), input_dim), DType::F32, &device).unwrap();
    let bias = Tensor::zeros(horizons.len(), DType::F32, &device).unwrap();
    tensors.insert("multi_horizon.weight".to_string(), weight);
    tensors.insert("multi_horizon.bias".to_string(), bias);

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let head = DynamicMultiHorizonHead::new(vb, input_dim, horizons.clone()).unwrap();

    let features = vec![0.5f32; input_dim];
    let forecasts = head.predict_horizons(&features, &device).unwrap();

    assert_eq!(forecasts.len(), 7);
    for (i, (h, val)) in forecasts.iter().enumerate() {
        assert_eq!(*h, horizons[i]);
        assert_eq!(*val, 0.0f32);
    }
}

#[test]
fn test_monotonic_quantile_head_non_crossing_guarantee() {
    let device = Device::Cpu;
    let quantiles = vec![0.05f32, 0.25, 0.50, 0.75, 0.95];
    let input_dim = 4;

    let mut tensors = HashMap::new();
    // Base linear
    tensors.insert(
        "quantile_base.weight".to_string(),
        Tensor::from_slice(&[0.1f32, -0.2, 0.3, -0.4], (1, input_dim), &device).unwrap(),
    );
    tensors.insert(
        "quantile_base.bias".to_string(),
        Tensor::from_slice(&[0.0f32], 1, &device).unwrap(),
    );

    // Delta linears
    for i in 1..quantiles.len() {
        tensors.insert(
            format!("quantile_delta_{i}.weight"),
            Tensor::from_slice(&[0.05f32 * i as f32; 4], (1, input_dim), &device).unwrap(),
        );
        tensors.insert(
            format!("quantile_delta_{i}.bias"),
            Tensor::from_slice(&[0.1f32], 1, &device).unwrap(),
        );
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let head = MonotonicQuantileHead::new(vb, input_dim, quantiles).unwrap();

    // Test on 20 diverse random inputs to guarantee monotonicity
    for step in 0..20 {
        let x = [
            (step as f32 * 0.7).sin() * 5.0,
            (step as f32 * 1.3).cos() * 3.0,
            (step as f32 * 0.4).sin() * 2.0,
            (step as f32 * 2.1).cos() * 4.0,
        ];

        let results = head.predict_quantiles(&x, &device).unwrap();
        assert_eq!(results.len(), 5);

        for j in 1..results.len() {
            let prev_q = results[j - 1].1;
            let curr_q = results[j].1;
            assert!(
                curr_q >= prev_q,
                "Quantile crossing violated at step {step}, index {j}: q_{} = {prev_q} > q_{} = {curr_q}",
                results[j - 1].0,
                results[j].0
            );
        }
    }

    // Pinball loss check
    let loss_under = MonotonicQuantileHead::pinball_loss(0.9, 10.0, 8.0);
    assert!((loss_under - 1.8).abs() < 1e-4);
    let loss_over = MonotonicQuantileHead::pinball_loss(0.9, 8.0, 10.0);
    assert!((loss_over - 0.2).abs() < 1e-4);
}

#[test]
fn test_velocity_flow_head() {
    let device = Device::Cpu;
    let dim = 3;
    let num_freqs = 4;
    let hidden_dim = 8;

    let mut tensors = HashMap::new();
    let time_dim = num_freqs * 2;
    tensors.insert(
        "flow_time_proj.weight".to_string(),
        Tensor::zeros((hidden_dim, time_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_time_proj.bias".to_string(),
        Tensor::zeros(hidden_dim, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_hidden.weight".to_string(),
        Tensor::zeros((hidden_dim, dim + hidden_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_hidden.bias".to_string(),
        Tensor::zeros(hidden_dim, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_out.weight".to_string(),
        Tensor::zeros((dim, hidden_dim), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "flow_out.bias".to_string(),
        Tensor::zeros(dim, DType::F32, &device).unwrap(),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let flow_head = VelocityFlowHead::new(vb, dim, num_freqs, hidden_dim).unwrap();

    let x = [0.1f32, 0.2, 0.3];
    let v0 = flow_head.predict_velocity(&x, 0.0, &device).unwrap();
    let v1 = flow_head.predict_velocity(&x, 1.0, &device).unwrap();
    assert_eq!(v0.len(), dim);
    assert_eq!(v1.len(), dim);
}

#[test]
fn test_loaded_model_weights_and_dequantization() {
    let device = Device::Cpu;
    let mut tensors = HashMap::new();
    let int8_data = [10i8, -20, 30, -40];
    let u8_bytes: Vec<u8> = int8_data.iter().map(|&x| x as u8).collect();
    let int8_tensor = Tensor::from_slice(&u8_bytes, (2, 2), &device).unwrap();
    tensors.insert("quant_layer.weight".to_string(), int8_tensor.clone());

    let weights = LoadedModelWeights {
        tensors,
        metadata: HashMap::new(),
        device: device.clone(),
    };

    assert_eq!(weights.len(), 1);
    assert!(!weights.is_empty());
    assert!(weights.contains_tensor("quant_layer.weight"));
    assert!(weights.memory_size_bytes() > 0);

    let dequant = LoadedModelWeights::dequantize_int8(&int8_tensor, 0.05).unwrap();
    assert_eq!(dequant.dtype(), DType::F32);
    let vals: Vec<f32> = dequant.flatten_all().unwrap().to_vec1().unwrap();
    assert!((vals[0] - 0.5).abs() < 1e-4);
    assert!((vals[2] - 1.5).abs() < 1e-4);
}
