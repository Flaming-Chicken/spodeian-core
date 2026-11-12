//! Rigour Plan Phases P3, P4, and P6: Comprehensive Robustness, Ablation,
//! Dtype Matrix, and Symplectic Long-Horizon Integration Tests.

use candle_core::{DType, Device, Result, Tensor, Var};
use sml_runtime::optim::{
    OptimizationRegime, Optimizer, SelfOptimizingMeta, SelfOptimizingMetaConfig,
};
use sml_runtime::precision::NumericType;
use sml_runtime::solvers::SymplecticPreservingIntegrator;

// ============================================================================
// Phase P3: Constant Perturbation Sensitivity Sweeps (+-50%)
// ============================================================================

#[test]
fn test_constant_sensitivity_sweeps() -> Result<()> {
    let dev = Device::Cpu;

    // Sweep curvature damping gamma in [0.25, 0.75] (+-50% around nominal 0.5)
    let damping_values = [0.25, 0.40, 0.50, 0.60, 0.75];
    for &damping in &damping_values {
        let x = Var::from_slice(&[5.0f64, -3.0f64], (2,), &dev)?;
        let config = SelfOptimizingMetaConfig {
            lr: 0.1,
            beta: 0.85,
            curvature_damping: damping,
            weight_decay: 0.0,
            warmup_steps: 2,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: OptimizationRegime::NonConvexValley,
            ..Default::default()
        };
        let mut opt = SelfOptimizingMeta::new(vec![x.clone()], config);

        for _ in 0..30 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let final_l = x.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
        assert!(
            final_l < 1e-4,
            "Curvature damping {} failed to converge: final loss = {}",
            damping,
            final_l
        );
    }

    // Sweep Polyak momentum beta in [0.5, 0.95]
    let beta_values = [0.50, 0.70, 0.85, 0.90, 0.95];
    for &beta in &beta_values {
        let x = Var::from_slice(&[4.0f64, 4.0f64], (2,), &dev)?;
        let config = SelfOptimizingMetaConfig {
            lr: 0.05,
            beta,
            curvature_damping: 0.5,
            weight_decay: 0.0,
            warmup_steps: 2,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: OptimizationRegime::NonConvexValley,
            ..Default::default()
        };
        let mut opt = SelfOptimizingMeta::new(vec![x.clone()], config);

        for _ in 0..40 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let final_l = x.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
        assert!(
            final_l < 1e-3,
            "Polyak beta {} failed to converge: final loss = {}",
            beta,
            final_l
        );
    }

    Ok(())
}

// ============================================================================
// Phase P4: Dtype Precision Matrix (F64, F32, BF16, F16)
// ============================================================================

#[test]
fn test_dtype_precision_matrix_convergence() -> Result<()> {
    let dev = Device::Cpu;

    // 1. F64 High-Precision
    {
        let x = Var::from_slice(&[3.0f64, -2.0f64], (2,), &dev)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F64),
        );
        for _ in 0..50 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let l = x.as_tensor().sqr()?.sum_all()?.to_scalar::<f64>()?;
        assert!(l < 1e-4, "F64 failed convergence: {}", l);
    }

    // 2. F32 Standard Precision
    {
        let x = Var::from_slice(&[3.0f32, -2.0f32], (2,), &dev)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F32),
        );
        for _ in 0..50 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let l = x.as_tensor().sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(l < 1e-3, "F32 failed convergence: {}", l);
    }

    // 3. BF16 Half Precision
    {
        let t = Tensor::from_slice(&[3.0f32, -2.0f32], (2,), &dev)?.to_dtype(DType::BF16)?;
        let x = Var::from_tensor(&t)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::BF16),
        );
        for _ in 0..60 {
            let loss = x.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let l = x.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(l < 0.05, "BF16 failed convergence: {}", l);
        assert_eq!(x.as_tensor().dtype(), DType::BF16, "BF16 dtype was mutated!");
    }

    // 4. F16 Half Precision
    {
        let t = Tensor::from_slice(&[3.0f32, -2.0f32], (2,), &dev)?.to_dtype(DType::F16)?;
        let x = Var::from_tensor(&t)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F16),
        );
        for _ in 0..60 {
            let loss = x.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let l = x.as_tensor().to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(l < 0.05, "F16 failed convergence: {}", l);
        assert_eq!(x.as_tensor().dtype(), DType::F16, "F16 dtype was mutated!");
    }

    // 5. Calibrated Guards Verification
    let (eff_f64, guard_f64) = SelfOptimizingMetaConfig::calibrate_guards(DType::F64, 1e-8);
    assert!(eff_f64 <= 1e-12);
    assert_eq!(guard_f64, 1e-12);

    let (eff_f32, guard_f32) = SelfOptimizingMetaConfig::calibrate_guards(DType::F32, 1e-8);
    assert!(eff_f32 >= 1e-8);
    assert_eq!(guard_f32, 1e-6);

    let (eff_bf16, guard_bf16) = SelfOptimizingMetaConfig::calibrate_guards(DType::BF16, 1e-8);
    assert!(eff_bf16 >= 1e-6);
    assert_eq!(guard_bf16, 1e-4);

    let (eff_f16, guard_f16) = SelfOptimizingMetaConfig::calibrate_guards(DType::F16, 1e-8);
    assert!(eff_f16 >= 1e-3);
    assert_eq!(guard_f16, 1e-3);

    Ok(())
}

// ============================================================================
// Phase P3: Leave-One-Out Structural Component Ablations
// ============================================================================

#[test]
fn test_structural_leave_one_out_ablations() -> Result<()> {
    let dev = Device::Cpu;

    // 1. Full Optimizer baseline on stiff Rosenbrock valley:
    let x_full = Var::from_slice(&[-1.2f64, 1.0f64], (2,), &dev)?;
    let mut opt_full = SelfOptimizingMeta::new(
        vec![x_full.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.02,
            beta: 0.85,
            curvature_damping: 0.5,
            weight_decay: 0.0,
            warmup_steps: 3,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: OptimizationRegime::NonConvexValley,
            ..Default::default()
        },
    );
    for _ in 0..100 {
        let x0 = x_full.as_tensor().get(0)?;
        let x1 = x_full.as_tensor().get(1)?;
        let term1 = (1.0 - &x0)?.sqr()?;
        let term2 = (100.0 * (&x1 - &x0.sqr()?)?.sqr()?)?;
        let loss = (&term1 + &term2)?;
        let grads = loss.backward()?;
        opt_full.step(&grads)?;
    }
    let full_loss = {
        let v = x_full.as_tensor().to_vec1::<f64>()?;
        (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2)
    };

    // 2. Ablation: No Curvature Damping / No Hill Gate (gamma = 0.0)
    let x_no_damp = Var::from_slice(&[-1.2f64, 1.0f64], (2,), &dev)?;
    let mut opt_no_damp = SelfOptimizingMeta::new(
        vec![x_no_damp.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.02,
            beta: 0.85,
            curvature_damping: 0.0,
            weight_decay: 0.0,
            warmup_steps: 3,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: OptimizationRegime::NonConvexValley,
            ..Default::default()
        },
    );
    for _ in 0..100 {
        let x0 = x_no_damp.as_tensor().get(0)?;
        let x1 = x_no_damp.as_tensor().get(1)?;
        let term1 = (1.0 - &x0)?.sqr()?;
        let term2 = (100.0 * (&x1 - &x0.sqr()?)?.sqr()?)?;
        let loss = (&term1 + &term2)?;
        let grads = loss.backward()?;
        opt_no_damp.step(&grads)?;
    }
    let no_damp_loss = {
        let v = x_no_damp.as_tensor().to_vec1::<f64>()?;
        (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2)
    };

    // 3. Ablation: No Polyak Schedule-Free Averaging (beta = 0.0)
    let x_no_polyak = Var::from_slice(&[-1.2f64, 1.0f64], (2,), &dev)?;
    let mut opt_no_polyak = SelfOptimizingMeta::new(
        vec![x_no_polyak.clone()],
        SelfOptimizingMetaConfig {
            lr: 0.02,
            beta: 0.0,
            curvature_damping: 0.5,
            weight_decay: 0.0,
            warmup_steps: 3,
            min_step_scale: 0.01,
            eps: 1e-12,
            regime: OptimizationRegime::NonConvexValley,
            ..Default::default()
        },
    );
    for _ in 0..100 {
        let x0 = x_no_polyak.as_tensor().get(0)?;
        let x1 = x_no_polyak.as_tensor().get(1)?;
        let term1 = (1.0 - &x0)?.sqr()?;
        let term2 = (100.0 * (&x1 - &x0.sqr()?)?.sqr()?)?;
        let loss = (&term1 + &term2)?;
        let grads = loss.backward()?;
        opt_no_polyak.step(&grads)?;
    }
    let no_polyak_loss = {
        let v = x_no_polyak.as_tensor().to_vec1::<f64>()?;
        (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2)
    };

    println!(
        "Ablations on Rosenbrock: Full = {:.4e} | NoDamping = {:.4e} | NoPolyak = {:.4e}",
        full_loss, no_damp_loss, no_polyak_loss
    );
    assert!(
        full_loss.is_finite() && no_damp_loss.is_finite() && no_polyak_loss.is_finite(),
        "Ablations produced non-finite values!"
    );

    Ok(())
}

// ============================================================================
// Phase P4: Numerical Edge Cases (Shock, Underflow, Stationary, 1D)
// ============================================================================

#[test]
fn test_numerical_edge_cases_robustness() -> Result<()> {
    let dev = Device::Cpu;

    // Edge 1: Exactly zero gradient (Stationary point must not move or produce NaN)
    {
        let x = Var::from_slice(&[0.0f64, 0.0f64], (2,), &dev)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F64),
        );
        let loss = x.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
        let p = x.as_tensor().to_vec1::<f64>()?;
        assert_eq!(p, vec![0.0, 0.0], "Stationary point perturbed by zero grad!");
    }

    // Edge 2: Extreme Gradient Shock (10^8 gradient spike must be bounded by Hill gate)
    {
        let x = Var::from_slice(&[1.0f64, 1.0f64], (2,), &dev)?;
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F64),
        );
        // Force shock step
        let shock_loss = (x.as_tensor().sqr()? * 1e8)?.sum_all()?;
        let grads = shock_loss.backward()?;
        opt.step(&grads)?;
        let p = x.as_tensor().to_vec1::<f64>()?;
        assert!(p[0].is_finite() && p[1].is_finite(), "Shock produced non-finite values!");
        assert!(
            p[0].abs() < 10.0 && p[1].abs() < 10.0,
            "Hill gate failed to bound 10^8 gradient shock: {:?}",
            p
        );
    }

    // Edge 3: Single scalar parameter (d = 1)
    {
        let x = Var::from_slice(&[10.0f64], (1,), &dev)?;
        let initial_val = x.as_tensor().to_vec1::<f64>()?[0];
        let mut opt = SelfOptimizingMeta::new(
            vec![x.clone()],
            SelfOptimizingMetaConfig::for_dtype(DType::F64),
        );
        for _ in 0..80 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            let grads = loss.backward()?;
            opt.step(&grads)?;
        }
        let val = x.as_tensor().to_vec1::<f64>()?[0];
        assert!(val.abs() < initial_val * 0.05, "Scalar 1D optimization failed: {}", val);
    }

    Ok(())
}

// ============================================================================
// Phase P6: Symplectic Integrator Long-Horizon Kepler Conservation (1,000 steps)
// ============================================================================

#[test]
fn test_long_horizon_kepler_energy_conservation() {
    let symp = SymplecticPreservingIntegrator::default().with_numeric_type(NumericType::F64);

    let e = 0.5f64;
    let mut q_unproj = vec![1.0 - e, 0.0];
    let mut p_unproj = vec![0.0, ((1.0 + e) / (1.0 - e)).sqrt()];
    let h = 0.005f64;

    let kepler_force = |q: &[f64]| -> Vec<f64> {
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let r3 = r * r * r;
        vec![-q[0] / r3, -q[1] / r3]
    };

    let kepler_hamiltonian = |q: &[f64], p: &[f64]| -> f64 {
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        0.5 * (p[0] * p[0] + p[1] * p[1]) - 1.0 / r
    };

    let target_energy = kepler_hamiltonian(&q_unproj, &p_unproj);
    let mut max_unprojected_drift = 0.0f64;

    // 1. Test unprojected Störmer-Verlet over 1,000 steps: bounded secular drift
    for _ in 0..1000 {
        let (next_q, next_p, energy) = symp.step_unprojected_f64(&q_unproj, &p_unproj, h, kepler_force, kepler_hamiltonian);
        q_unproj = next_q;
        p_unproj = next_p;
        let drift = (energy - target_energy).abs();
        if drift > max_unprojected_drift {
            max_unprojected_drift = drift;
        }
    }

    println!("Kepler 1000-step unprojected f64 max energy drift: {:.2e}", max_unprojected_drift);
    assert!(
        max_unprojected_drift < 5e-3,
        "Symplectic area-preservation failed: unprojected drift = {:.2e}",
        max_unprojected_drift
    );

    // 2. Test projected step in f64: maintains energy to high precision
    let mut q_proj = vec![1.0 - e, 0.0];
    let mut p_proj = vec![0.0, ((1.0 + e) / (1.0 - e)).sqrt()];
    let mut max_projected_drift = 0.0f64;
    for _ in 0..1000 {
        let (next_q, next_p, energy) = symp.step_f64(&q_proj, &p_proj, h, target_energy, kepler_force, kepler_hamiltonian);
        q_proj = next_q;
        p_proj = next_p;
        let drift = (energy - target_energy).abs();
        if drift > max_projected_drift {
            max_projected_drift = drift;
        }
    }

    println!("Kepler 1000-step projected f64 max energy drift: {:.2e}", max_projected_drift);
    assert!(
        max_projected_drift < 1e-4,
        "Projected symplectic drift too high: {:.2e}",
        max_projected_drift
    );
}
