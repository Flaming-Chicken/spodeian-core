//! Integration regression tests for SelfOptimizingMeta and optimizer special cases.

use candle_core::{Device, Result, Tensor, Var};
use sml_runtime::optim::{
    Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig,
};

#[test]
fn test_regression_ultra_stiff_ravine() -> Result<()> {
    let device = Device::Cpu;
    // Stiff ravine with condition number kappa = 10^4
    let init_xy = vec![2.0f32, 0.1f32];
    let xy = Var::from_slice(&init_xy, (2,), &device)?;
    let stiffness = Tensor::from_slice(&[1.0f32, 10000.0f32], (2,), &device)?;

    let mut opt = SelfOptimizingMeta::new(
        vec![xy.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.05,
            beta: 0.85,
            curvature_damping: 0.8,
            weight_decay: 0.0,
            warmup_steps: 2,
            min_step_scale: 0.01,
            eps: 1e-8,
            regime: sml_runtime::optim::OptimizationRegime::Auto,
            ..Default::default()
        },
    );


    let calc_loss = |var: &Var| -> Result<f32> {
        let sq = var.as_tensor().sqr()?;
        let prod = sq.broadcast_mul(&stiffness)?;
        let l = (prod.sum_all()?.to_vec0::<f32>()?) * 0.5;
        Ok(l)
    };

    let l0 = calc_loss(&xy)?;
    assert!(l0 > 1.0, "Initial loss should be > 1.0");

    for _s in 0..50 {
        let sq = xy.as_tensor().sqr()?;
        let prod = sq.broadcast_mul(&stiffness)?;
        let loss = (prod.sum_all()? * 0.5)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let l_final = calc_loss(&xy)?;
    assert!(
        !l_final.is_nan() && !l_final.is_infinite(),
        "Loss became NaN or infinite!"
    );
    assert!(
        l_final < l0 * 0.1,
        "Meta optimizer should reduce stiff loss by >90%: init = {}, final = {}",
        l0,
        l_final
    );
    Ok(())
}

#[test]
fn test_regression_smooth_operators_differentiability() -> Result<()> {
    let device = Device::Cpu;
    // Verify that zero or tiny gradients do not cause division by zero or NaN
    let x = Var::from_slice(&[0.0f32, 0.0f32, 0.0f32], (3,), &device)?;
    let mut opt = SelfOptimizingMeta::new(
        vec![x.clone()],
        SelfOptimizingMetaConfig::default(),
    );

    let loss = x.as_tensor().sqr()?.sum_all()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;

    let updated = x.as_tensor().to_vec1::<f32>()?;
    for val in updated {
        assert!(!val.is_nan() && !val.is_infinite(), "Encountered NaN in smooth zero gradient step");
        assert_eq!(val, 0.0, "Zero gradient should yield zero displacement at origin");
    }
    Ok(())
}

#[test]
fn test_regression_coordinate_disparity_handling() -> Result<()> {
    let device = Device::Cpu;
    // 1 steep coordinate (1000 * x0^2) vs 1 flat coordinate (x1^2)
    let xy = Var::from_slice(&[1.0f32, 1.0f32], (2,), &device)?;
    let mut opt = SelfOptimizingMeta::new(
        vec![xy.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.05,

            beta: 0.85,
            curvature_damping: 0.5,
            weight_decay: 0.0,
            warmup_steps: 2,
            min_step_scale: 0.1,
            eps: 1e-8,
            regime: sml_runtime::optim::OptimizationRegime::Auto,
            ..Default::default()
        },
    );

    let calc_loss = |v: &Var| -> Result<f32> {
        let x = v.narrow(0, 0, 1)?;
        let y = v.narrow(0, 1, 1)?;
        let t1 = (x.sqr()? * 1000.0)?;
        let t2 = y.sqr()?;
        let l = (&t1 + &t2)?.sum_all()?.to_vec0::<f32>()?;
        Ok(l)
    };

    let l0 = calc_loss(&xy)?;
    for _ in 0..50 {
        let x = xy.narrow(0, 0, 1)?;
        let y = xy.narrow(0, 1, 1)?;
        let t1 = (x.sqr()? * 1000.0)?;
        let t2 = y.sqr()?;
        let loss = (&t1 + &t2)?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }

    let l_final = calc_loss(&xy)?;
    assert!(
        l_final < l0 * 0.1,
        "Loss should reduce by at least 90%: init = {}, final = {}",
        l0,
        l_final
    );
    Ok(())
}

#[test]
fn test_multi_dtype_and_matrix_polar_optimization() -> Result<()> {
    use candle_core::DType;
    let device = Device::Cpu;

    // 1. Test F64 2D matrix optimization (4 Newton-Schulz iterations + F64 epsilon floor)
    let w_f64 = Var::from_tensor(&Tensor::from_slice(
        &[2.0f64, -1.0, 0.5, 1.5, -2.0, 1.0],
        (2, 3),
        &device,
    )?)?;
    let mut opt_f64 = SelfOptimizingMeta::new(
        vec![w_f64.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.08,
            ..SelfOptimizingMetaConfig::for_dtype(DType::F64)
        },
    );
    let l0_f64 = w_f64.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
    for _ in 0..25 {
        let loss = w_f64.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt_f64.step(&grads)?;
    }
    let lf_f64 = w_f64.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
    assert!(
        lf_f64 < l0_f64 * 0.1,
        "F64 2D matrix loss should decrease by >90%: {} -> {}",
        l0_f64,
        lf_f64
    );

    // 2. Test F32 2D matrix optimization (3 Newton-Schulz iterations)
    let w_f32 = Var::from_tensor(&Tensor::from_slice(
        &[2.0f32, -1.0, 0.5, 1.5, -2.0, 1.0],
        (2, 3),
        &device,
    )?)?;
    let mut opt_f32 = SelfOptimizingMeta::new(
        vec![w_f32.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.08,
            ..SelfOptimizingMetaConfig::for_dtype(DType::F32)
        },
    );
    let l0_f32 = w_f32.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
    for _ in 0..25 {
        let loss = w_f32.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt_f32.step(&grads)?;
    }
    let lf_f32 = w_f32.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
    assert!(
        lf_f32 < l0_f32 * 0.1,
        "F32 2D matrix loss should decrease by >90%: {} -> {}",
        l0_f32,
        lf_f32
    );

    Ok(())
}
