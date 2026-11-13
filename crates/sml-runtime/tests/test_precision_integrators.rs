//! Unit & Integration tests for Type-Agnostic, Precision-Aware Integrators
//! Testing f64, f32, bf16, f16, Posit32, and Information-Theoretic Gating.

use sml_runtime::precision::{
    InformationRateGatingPolicy, NumericType, compute_scaled_error_norm, get_precision,
};
use sml_runtime::solvers::{SelfOptimizingMetaIntegrator, SymplecticPreservingIntegrator};

#[test]
fn test_precision_scaling_f64_vs_f32_vs_bf16_vs_posit() {
    let p_f64 = get_precision(NumericType::F64);
    let p_f32 = get_precision(NumericType::F32);
    let p_bf16 = get_precision(NumericType::Bf16);
    let p_f16 = get_precision(NumericType::F16);
    let p_p32 = get_precision(NumericType::Posit32);

    // Verify machine epsilon ordering
    assert!(p_f64.machine_epsilon(None) < p_p32.machine_epsilon(Some(1.0)));
    assert!(p_p32.machine_epsilon(Some(1.0)) < p_f32.machine_epsilon(None));
    assert!(p_f32.machine_epsilon(None) < p_f16.machine_epsilon(None));
    assert!(p_f16.machine_epsilon(None) < p_bf16.machine_epsilon(None));

    // Verify energy drift guards are correctly scaled
    assert!(p_f64.energy_drift_guard() < 1e-14);
    assert!(p_f32.energy_drift_guard() >= 1e-8);
    assert!(p_bf16.energy_drift_guard() >= 5e-3);
    assert!(p_p32.energy_drift_guard() < 1e-7);
}

#[test]
fn test_scale_free_error_norm_invariance() {
    // When state and error are scaled by 1,000x (e.g. meters to millimeters),
    // the relative scaled error norm should remain strictly invariant!
    let err_m = [1e-3, 2e-3];
    let x_m = [1.0, 2.0];
    let norm_m = compute_scaled_error_norm(&err_m, &x_m, &x_m, 1e-4, 1e-3);

    let err_mm = [1.0, 2.0]; // 1000x larger
    let x_mm = [1000.0, 2000.0]; // 1000x larger
    let norm_mm = compute_scaled_error_norm(&err_mm, &x_mm, &x_mm, 1e-1, 1e-3);

    assert!(
        (norm_m - norm_mm).abs() < 1e-4,
        "Scale-free norm must be unit-invariant!"
    );
}

#[test]
fn test_information_theoretic_gating_behavior() {
    // Scenario 1: Heavy Neural Network (10 ms latency)
    let mut heavy_policy = InformationRateGatingPolicy::new(1e4); // 10,000 bits/sec
    heavy_policy.eval_latency_ema_sec = 0.010; // 10 ms

    // Low curvature / small correction = 0.1 bits of information
    // Rate = 0.1 / 0.010 = 10 bits/sec << 10,000 bits/sec -> Must skip secondary stage!
    assert!(!heavy_policy.should_evaluate_secondary_stage(0.1));

    // High curvature / large transient = 150 bits of information
    // Rate = 150 / 0.010 = 15,000 bits/sec > 10,000 bits/sec -> Must evaluate secondary stage!
    assert!(heavy_policy.should_evaluate_secondary_stage(150.0));

    // Scenario 2: Ultra-cheap vector field (100 ns latency)
    let mut cheap_policy = InformationRateGatingPolicy::new(1e4);
    cheap_policy.eval_latency_ema_sec = 1e-7; // 100 ns
    // Even a tiny 0.06 bits gain yields 600,000 bits/sec >> 10,000 bits/sec -> Justified!
    assert!(cheap_policy.should_evaluate_secondary_stage(0.06));
}

#[test]
fn test_symplectic_integrator_precision_adaptation() {
    let symp_f64 = SymplecticPreservingIntegrator::default().with_numeric_type(NumericType::F64);
    let symp_bf16 = SymplecticPreservingIntegrator::default().with_numeric_type(NumericType::Bf16);

    let q = vec![1.0, 0.0];
    let p = vec![0.0, 1.0];
    let h = 0.01;
    let target_e = 1.0;

    // Harmonic oscillator: force = -q, Hamiltonian = 0.5 * (p^2 + q^2)
    let force = |q: &[f32]| vec![-q[0], -q[1]];
    let ham = |q: &[f32], p: &[f32]| 0.5 * (p[0] * p[0] + p[1] * p[1] + q[0] * q[0] + q[1] * q[1]);

    let (_, _, e_f64) = symp_f64.step(&q, &p, h, target_e, force, ham);
    let (_, _, e_bf16) = symp_bf16.step(&q, &p, h, target_e, force, ham);

    // Both should maintain bounded energy, but f64 precision guard ensures high-fidelity correction
    assert!((e_f64 - target_e).abs() < 0.01);
    assert!((e_bf16 - target_e).abs() < 0.05);
}

#[test]
fn test_self_optimizing_meta_integrator_with_different_precisions() {
    for num_type in [
        NumericType::F64,
        NumericType::F32,
        NumericType::Posit32,
        NumericType::Bf16,
    ] {
        let meta = SelfOptimizingMetaIntegrator::default().with_numeric_type(num_type);
        let x0 = vec![1.0, 0.0];
        let (final_x, diagnostics, nfe) = meta.solve_auto(&x0, |x, _t| vec![-x[1], x[0]]);

        assert_eq!(final_x.len(), 2);
        assert!(!diagnostics.is_empty());
        assert!(nfe > 0);
        assert!(final_x[0].is_finite() && final_x[1].is_finite());
    }
}
