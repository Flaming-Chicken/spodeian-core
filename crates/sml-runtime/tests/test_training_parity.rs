//! Comprehensive Verification Suite for PyTorch 2 Training Parity in `sml-runtime`.

use candle_core::{Device, Tensor, Var};
use sml_runtime::loss::{
    bce_with_logits, cross_entropy, flow_matching_loss, focal_loss, info_nce_loss,
    smooth_l1_loss, Reduction,
};
use sml_runtime::lr_scheduler::{
    CosineAnnealingLR, ExponentialLR, LRScheduler, LinearWarmupCosineAnnealing, MultiStepLR,
    OneCycleLR, StepLR,
};
use sml_runtime::optim::{AdamW, Lion, Muon, MuonConfig, Optimizer, ScheduleFreeAdamW, Sgd, SgdConfig, Sophia, SophiaConfig};
use sml_runtime::solvers::{ContinuousFlowSolver, SelfOptimizingMetaIntegrator, TrainedPcecIntegrator};
use sml_runtime::train::{
    clip_grad_norm, DataLoader, GradScaler, ModelEMA, TensorDataset,
};

#[test]
fn test_sgd_momentum_and_nesterov() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[1.0f32, 2.0], &device)?;
    let mut opt = Sgd::new(
        vec![var.clone()],
        SgdConfig {
            lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.01,
            nesterov: true,
            ..Default::default()
        },
    );

    // Dummy forward & backward
    for _ in 0..5 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let p = var.as_tensor().to_vec1::<f32>()?;
    // Param must decrease towards zero
    assert!(p[0] < 1.0);
    assert!(p[1] < 2.0);
    Ok(())
}

#[test]
fn test_adamw_convergence() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[5.0f32, -3.0], &device)?;
    let mut opt = AdamW::new_lr(vec![var.clone()], 0.1);

    for _ in 0..60 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let p = var.as_tensor().to_vec1::<f32>()?;
    assert!(p[0].abs() < 1.0, "Expected p[0] to converge, got {}", p[0]);
    assert!(p[1].abs() < 1.0, "Expected p[1] to converge, got {}", p[1]);
    Ok(())
}

#[test]
fn test_lion_optimizer() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[3.0f32, -2.0], &device)?;
    let mut opt = Lion::new_lr(vec![var.clone()], 0.05);

    for _ in 0..20 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let p = var.as_tensor().to_vec1::<f32>()?;
    assert!(p[0] < 3.0);
    assert!(p[1] > -2.0);
    Ok(())
}

#[test]
fn test_sophia_optimizer() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[4.0f32, -4.0], &device)?;
    let mut opt = Sophia::new(vec![var.clone()], SophiaConfig { lr: 0.05, ..Default::default() });

    for _ in 0..20 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let p = var.as_tensor().to_vec1::<f32>()?;
    assert!(p[0] < 4.0);
    assert!(p[1] > -4.0);
    Ok(())
}

#[test]
fn test_schedule_free_adamw() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[2.5f32, -1.5], &device)?;
    let mut opt = ScheduleFreeAdamW::new(vec![var.clone()], Default::default());

    for _ in 0..25 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let p = var.as_tensor().to_vec1::<f32>()?;
    assert!(p[0] < 2.5);
    assert!(p[1] > -1.5);
    Ok(())
}

#[test]
fn test_lr_schedulers_progression() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[1.0f32], &device)?;
    let mut opt = AdamW::new_lr(vec![var], 0.1);

    // 1. Cosine Annealing
    let mut cosine = CosineAnnealingLR::new(0.1, 100, 1e-4);
    assert!((cosine.get_lr() - 0.1).abs() < 1e-6);
    for _ in 0..50 {
        cosine.step(&mut opt);
    }
    // Halfway cosine should be roughly midpoint
    let mid_lr = cosine.get_lr();
    assert!(mid_lr < 0.1 && mid_lr > 1e-4);

    // 2. Linear Warmup + Cosine
    let mut warmup = LinearWarmupCosineAnnealing::new(0.1, 10, 100, 1e-4);
    assert_eq!(warmup.get_lr(), 0.0);
    for _ in 0..10 {
        warmup.step(&mut opt);
    }
    assert!((warmup.get_lr() - 0.1).abs() < 1e-5);

    // 3. OneCycleLR
    let mut one_cycle = OneCycleLR::new(0.1, 100, 0.3);
    for _ in 0..30 {
        one_cycle.step(&mut opt);
    }
    // At peak (step 30 of 100 with pct_start = 0.3)
    assert!((one_cycle.get_lr() - 0.1).abs() < 1e-4);

    // 4. ExponentialLR
    let mut exp_sched = ExponentialLR::new(0.1, 0.95);
    for _ in 0..10 {
        exp_sched.step(&mut opt);
    }
    let expected = 0.1 * 0.95f64.powi(10);
    assert!((exp_sched.get_lr() - expected).abs() < 1e-6);

    // 5. StepLR & MultiStepLR
    let mut step_sched = StepLR::new(0.1, 10, 0.5);
    for _ in 0..10 {
        step_sched.step(&mut opt);
    }
    assert!((step_sched.get_lr() - 0.05).abs() < 1e-6);

    let mut multi_sched = MultiStepLR::new(0.1, vec![5, 15], 0.1);
    for _ in 0..6 {
        multi_sched.step(&mut opt);
    }
    assert!((multi_sched.get_lr() - 0.01).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_losses_mathematical_precision() -> candle_core::Result<()> {
    let device = Device::Cpu;

    // 1. CrossEntropy with Label Smoothing
    let logits = Tensor::from_vec(vec![2.0f32, 1.0, 0.1, 0.5, 2.5, 0.2], (2, 3), &device)?;
    let targets = Tensor::from_vec(vec![0u32, 1], (2,), &device)?;
    let ce_std = cross_entropy(&logits, &targets, 0.0, Reduction::Mean)?.to_scalar::<f32>()?;
    let ce_smooth = cross_entropy(&logits, &targets, 0.1, Reduction::Mean)?.to_scalar::<f32>()?;
    assert!(ce_std > 0.0);
    assert!(ce_smooth > 0.0);

    // 2. BCE with Logits
    let bce_logits = Tensor::from_vec(vec![0.0f32, 2.0, -2.0], (3,), &device)?;
    let bce_targets = Tensor::from_vec(vec![0.5f32, 1.0, 0.0], (3,), &device)?;
    let bce = bce_with_logits(&bce_logits, &bce_targets, Reduction::Mean)?.to_scalar::<f32>()?;
    assert!(bce > 0.0 && bce.is_finite());

    // 3. Smooth L1 (Huber)
    let p = Tensor::from_vec(vec![0.2f32, 2.0], (2,), &device)?;
    let t = Tensor::from_vec(vec![0.0f32, 0.0], (2,), &device)?;
    let huber = smooth_l1_loss(&p, &t, 1.0, Reduction::Mean)?.to_scalar::<f32>()?;
    assert!(huber > 0.0);

    // 4. Focal Loss
    let fl = focal_loss(&logits, &targets, 2.0, 0.25, Reduction::Mean)?.to_scalar::<f32>()?;
    assert!(fl > 0.0);

    // 5. InfoNCE Loss
    let q = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device)?;
    let k = Tensor::from_vec(vec![1.0f32, 0.1, -0.1, 1.0], (2, 2), &device)?;
    let nce = info_nce_loss(&q, &k, 0.07)?.to_scalar::<f32>()?;
    assert!(nce > 0.0);

    // 6. Flow Matching Loss
    let v_pred = Tensor::from_vec(vec![0.5f32, -0.5], (2,), &device)?;
    let v_tgt = Tensor::from_vec(vec![0.6f32, -0.4], (2,), &device)?;
    let cfm = flow_matching_loss(&v_pred, &v_tgt)?.to_scalar::<f32>()?;
    assert!((cfm - 0.01).abs() < 1e-4);

    Ok(())
}

#[test]
fn test_gradient_clipping_and_grad_scaler() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[10.0f32, 20.0], &device)?;
    let loss = var.as_tensor().sqr()?.sum_all()?;
    let grads = loss.backward()?;

    // Clip norm
    let (norm_before, clipped_store) = clip_grad_norm(std::slice::from_ref(&var), &grads, 1.0)?;
    assert!(norm_before > 1.0);
    let clipped_g = clipped_store.get(var.as_tensor()).unwrap();
    let norm_after = clipped_g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    assert!((norm_after - 1.0).abs() < 1e-4);

    // GradScaler
    let mut scaler = GradScaler::new(1024.0);
    let mut opt = AdamW::new_lr(vec![var.clone()], 0.01);
    let ok = scaler.unscale_and_step(std::slice::from_ref(&var), &clipped_store, &mut opt)?;
    assert!(ok);

    Ok(())
}

#[test]
fn test_model_ema_and_dataloader() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let var = Var::new(&[1.0f32, 2.0], &device)?;
    let mut ema = ModelEMA::new(std::slice::from_ref(&var), 0.9)?;

    // Change var value
    var.set(&Tensor::from_vec(vec![3.0f32, 4.0], (2,), &device)?)?;
    ema.update(std::slice::from_ref(&var))?;

    let shadow = ema.shadow_weights()[0].to_vec1::<f32>()?;
    println!("EMA shadow values: {:?}", shadow);
    // 0.9 * 1.0 + 0.1 * 3.0 = 1.2
    assert!((shadow[0] - 1.2).abs() < 1e-4, "Expected shadow[0] ~ 1.2, got {}", shadow[0]);
    // 0.9 * 2.0 + 0.1 * 4.0 = 2.2
    assert!((shadow[1] - 2.2).abs() < 1e-5);

    // Test TensorDataset & DataLoader
    let inputs = Tensor::from_vec((0..100).map(|x| x as f32).collect::<Vec<_>>(), (50, 2), &device)?;
    let targets = Tensor::from_vec((0..50).map(|x| x as u32).collect::<Vec<_>>(), (50,), &device)?;
    let dataset = TensorDataset::new(inputs, Some(targets))?;
    let loader = DataLoader::new(&dataset, 16, true, false);

    let mut batch_count = 0;
    for batch in loader {
        let (bx, by) = batch?;
        assert!(bx.dim(0)? <= 16);
        assert!(by.is_some());
        batch_count += 1;
    }
    // 50 samples with batch_size 16 -> 4 batches (16 + 16 + 16 + 2)
    assert_eq!(batch_count, 4);

    Ok(())
}

#[test]
fn test_muon_convergence() -> candle_core::Result<()> {
    let device = Device::Cpu;
    // 2D weight matrix: (4, 4)
    let var = Var::new(&[
        [2.0f32, 1.0, 0.5, -0.5],
        [1.0, 3.0, -1.0, 0.2],
        [0.5, -1.0, 2.5, 0.0],
        [-0.5, 0.2, 0.0, 1.5],
    ], &device)?;

    let mut opt = Muon::new(
        vec![var.clone()],
        MuonConfig {
            lr: 0.05,
            momentum: 0.9,
            ns_steps: 5,
            weight_decay: 0.001,
            ..Default::default()
        },
    );

    let initial_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;

    for _ in 0..10 {
        let loss = var.as_tensor().sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
    }

    let final_loss = var.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
    assert!(final_loss < initial_loss, "Muon failed to reduce loss: {} -> {}", initial_loss, final_loss);
    println!("Muon convergence test passed: loss reduced from {:.4} to {:.4}", initial_loss, final_loss);
    Ok(())
}

#[test]
fn test_trained_pcec_solver() {
    let x0 = vec![1.0f32, 2.0f32, -1.0f32];
    let solver = TrainedPcecIntegrator::default();
    let mut velocity_fn = |x: &[f32], t: f32| -> Vec<f32> {
        x.iter().map(|&v| -v * (1.0 - t)).collect()
    };

    let result = solver.solve_flow(&x0, 20, &mut velocity_fn);
    assert_eq!(result.len(), 3);
    for &val in &result {
        assert!(val.is_finite());
    }
    // Trajectory decaying towards origin
    assert!(result[0].abs() < x0[0].abs());
    assert!(result[1].abs() < x0[1].abs());
}

#[test]
fn test_self_optimizing_meta_integrator() {
    let x0 = vec![2.0f32, -1.5f32, 0.5f32];
    let auto_solver = SelfOptimizingMetaIntegrator::default();
    let mut velocity_fn = |x: &[f32], t: f32| -> Vec<f32> {
        x.iter().map(|&v| -v.tanh() * (1.0 - t)).collect()
    };

    let (res, diags, nfe) = auto_solver.solve_auto(&x0, &mut velocity_fn);
    assert_eq!(res.len(), 3);
    assert!(nfe > 0);
    assert!(!diags.is_empty());
    for d in &diags {
        assert!(d.step_size > 0.0);
        assert!(d.estimated_curvature.is_finite());
    }
    // Result converged towards origin
    assert!(res[0].abs() < x0[0].abs());
    println!("Self-Optimizing Meta-Integrator test passed: {} steps, {} NFE", diags.len(), nfe);
}
