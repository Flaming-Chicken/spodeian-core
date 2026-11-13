//! Comprehensive Verification Test Suite for All Numerical Types in sml-runtime.
//!
//! Validates:
//! 1. All 12 variants of `NumericType` (F64, F32, Tf32, F16, Bf16, Fp8E4M3, Fp8E5M2,
//!    Posit32, Posit16, Posit8, Log16, Log8) via `DataTypeSpec` and `NumericPrecision`.
//! 2. Dynamic machine epsilon models (constant for IEEE floats vs tapered for Gustafson Posits).
//! 3. Precision guards scaling (eff_eps, num_guard, temporal_guard, energy_drift_guard, etc.).
//! 4. Quantized types: NF4 (Voronoi decision tree), Int8, Ternary SIMD, BNN/Binary.
//! 5. Logarithmic number systems (Log8, Log16) dynamic range and exact multiplication.
//! 6. Discrete Umbral calculus types (i32, i16, i8) on integer lattices.
//! 7. Numerical stability of ODE solvers across every single `NumericType`.

use candle_core::DType;
use sml_runtime::logarithmic::{Log8, Log16};
use sml_runtime::optim::SelfOptimizingMetaConfig;
use sml_runtime::precision::{DataTypeSpec, NumericType, get_precision};
use sml_runtime::quantization::{
    NF4_CENTROIDS, PositDecoder, dequantize_nf4, quantize_to_nf4, ternary_matmul_simd_f32,
};
use sml_runtime::solvers::{SelfOptimizingMetaIntegrator, SymplecticPreservingIntegrator};
use sml_runtime::umbral::{RationalResidue, UmbralIntegerState, UmbralLatticeScalar};

// ============================================================================
// 1. All 12 NumericType Specifications & Guard Consistency
// ============================================================================

#[test]
fn test_all_12_numeric_type_specs() {
    let all_types = [
        NumericType::F64,
        NumericType::F32,
        NumericType::Tf32,
        NumericType::F16,
        NumericType::Bf16,
        NumericType::Fp8E4M3,
        NumericType::Fp8E5M2,
        NumericType::Posit32,
        NumericType::Posit16,
        NumericType::Posit8,
        NumericType::Log16,
        NumericType::Log8,
    ];

    for &nt in &all_types {
        let spec = DataTypeSpec::of(nt);
        let prec = get_precision(nt);

        assert_eq!(spec.numeric_type, nt);
        assert!(spec.bit_width > 0 && spec.bit_width <= 64);
        assert!(spec.storage_bytes >= 1 && spec.storage_bytes <= 8);
        assert!(spec.min_positive > 0.0);
        assert!(spec.max_normal > spec.min_positive);
        assert!(spec.default_eff_eps > 0.0);
        assert!(spec.default_num_guard > 0.0);
        assert!(spec.min_step_scale > 0.0 && spec.min_step_scale <= 1.0);

        // Guard bounds must be non-zero and positive
        assert!(spec.temporal_guard() > 0.0);
        assert!(spec.denominator_guard() > 0.0);
        assert!(spec.energy_drift_guard() > 0.0);
        assert!(spec.pid_antiwindup_guard() > 0.0);
        assert!(spec.adaptive_clamp_bound(1.0, 1.0) > 0.0);

        // Precision trait object must match spec
        let eps_golden = spec.machine_epsilon(None);
        assert_eq!(eps_golden, prec.machine_epsilon(None));
        assert!(prec.denominator_guard() > 0.0);
        assert!(prec.energy_drift_guard() > 0.0);
    }
}

// ============================================================================
// 2. Machine Epsilon Ordering & Tapered Posit Behavior
// ============================================================================

#[test]
fn test_machine_epsilon_ordering_and_tapering() {
    let spec_f64 = DataTypeSpec::of(NumericType::F64);
    let spec_f32 = DataTypeSpec::of(NumericType::F32);
    let spec_tf32 = DataTypeSpec::of(NumericType::Tf32);
    let spec_f16 = DataTypeSpec::of(NumericType::F16);
    let spec_bf16 = DataTypeSpec::of(NumericType::Bf16);
    let spec_fp8_e4m3 = DataTypeSpec::of(NumericType::Fp8E4M3);
    let spec_fp8_e5m2 = DataTypeSpec::of(NumericType::Fp8E5M2);

    // Standard IEEE precision ordering:
    // F64 < F32 < (Tf32 == F16) < Bf16 < Fp8E4M3 < Fp8E5M2
    assert!(spec_f64.machine_epsilon(None) < spec_f32.machine_epsilon(None));
    assert!(spec_f32.machine_epsilon(None) < spec_tf32.machine_epsilon(None));
    assert_eq!(
        spec_tf32.machine_epsilon(None),
        spec_f16.machine_epsilon(None)
    );
    assert!(spec_f16.machine_epsilon(None) < spec_bf16.machine_epsilon(None));
    assert!(spec_bf16.machine_epsilon(None) < spec_fp8_e4m3.machine_epsilon(None));
    assert!(spec_fp8_e4m3.machine_epsilon(None) < spec_fp8_e5m2.machine_epsilon(None));

    // Posit tapering verification:
    // Precision is maximized in golden zone (|x| ≈ 1.0) and tapers toward extreme values.
    let spec_p32 = DataTypeSpec::of(NumericType::Posit32);
    let spec_p16 = DataTypeSpec::of(NumericType::Posit16);
    let spec_p8 = DataTypeSpec::of(NumericType::Posit8);

    // Golden zone epsilons
    let eps_p32_golden = spec_p32.machine_epsilon_at(Some(1.0));
    let eps_p16_golden = spec_p16.machine_epsilon_at(Some(1.0));
    let eps_p8_golden = spec_p8.machine_epsilon_at(Some(1.0));

    assert!(eps_p32_golden < eps_p16_golden);
    assert!(eps_p16_golden < eps_p8_golden);

    // For Posit32, golden zone precision (2^-27) is superior to single-precision F32 (2^-23)!
    assert!(eps_p32_golden < spec_f32.machine_epsilon(None));

    // Tapering: At extreme magnitude (|x| = 1e6), regime bits consume fraction bits,
    // so machine epsilon must be larger (less precision) than at golden zone.
    let eps_p32_large = spec_p32.machine_epsilon_at(Some(1e6));
    assert!(
        eps_p32_large > eps_p32_golden,
        "Posit32 must taper at extreme values: golden={:.2e}, large={:.2e}",
        eps_p32_golden,
        eps_p32_large
    );

    let eps_p16_large = spec_p16.machine_epsilon_at(Some(1e4));
    assert!(eps_p16_large > eps_p16_golden);
}

// ============================================================================
// 3. Posit8 LUT & Decoding Behavior
// ============================================================================

#[test]
fn test_posit8_codec_and_lut() {
    let lut = PositDecoder::get_posit8_lut();
    assert_eq!(lut.len(), 256);
    assert_eq!(lut[0], 0.0);
    assert!(lut[0x80].is_nan()); // NaR (Not a Real)

    // Positive values must be strictly finite and positive
    let val_40 = PositDecoder::decode_posit8(0x40);
    assert!(val_40.is_finite() && val_40 > 0.0);
    assert_eq!(val_40, 1.0); // 0x40 in Posit<8,0> represents exactly 1.0
}

// ============================================================================
// 4. Logarithmic Arithmetic (Log8 & Log16) Exactness & Underflow Immunity
// ============================================================================

#[test]
fn test_logarithmic_arithmetic_exactness() {
    // Log8 multiplication via integer addition: a * b = 2^(log2(a) + log2(b))
    let l1 = Log8::from_f32(2.0);
    let l2 = Log8::from_f32(4.0);
    let l_prod = l1 * l2;
    let prod_f32 = l_prod.to_f32();
    assert!(
        (prod_f32 - 8.0).abs() < 1.0,
        "Log8 multiplication 2 * 4 failed: got {}",
        prod_f32
    );

    // Log16 wide dynamic range without underflow:
    let tiny1 = Log16::from_f32(1e-12);
    let tiny2 = Log16::from_f32(1e-10);
    let tiny_prod = tiny1 * tiny2;
    assert!(
        tiny_prod.to_f32() != 0.0,
        "Log16 must not underflow at 1e-22!"
    );
}

// ============================================================================
// 5. NF4 Voronoi Decision-Tree Quantization Bounds
// ============================================================================

#[test]
fn test_nf4_voronoi_quantization_bounds() {
    // Normal distribution quantiles in NF4 must cover [-1.0, 1.0] symmetrically
    assert_eq!(NF4_CENTROIDS.len(), 16);
    assert!((NF4_CENTROIDS[0] + 1.0).abs() < 1e-5);
    assert!((NF4_CENTROIDS[15] - 1.0).abs() < 1e-5);

    // Verify 4-level decision tree quantize_to_nf4 is monotonic and closest to centroid
    for i in 0..16 {
        let centroid = NF4_CENTROIDS[i];
        let idx = quantize_to_nf4(centroid) as usize;
        assert_eq!(
            idx, i,
            "Centroid {} should quantize exactly to index {}",
            centroid, i
        );

        let dequant = dequantize_nf4(idx as u8);
        assert_eq!(dequant, centroid);
    }
}

// ============================================================================
// 6. Ternary & Integer Lattice Umbral Types
// ============================================================================

#[test]
fn test_ternary_and_umbral_lattice_types() {
    // Ternary SIMD matmul: 4 weights per byte: 0x01 = +1, 0x03 = -1, 0x00 = 0
    let packed = vec![0b01_00_11_01]; // w0 = 1, w1 = -1, w2 = 0, w3 = 1
    let acts = vec![2.0, 3.0, 10.0, 5.0];
    let mut out = vec![0.0];
    ternary_matmul_simd_f32(&packed, &acts, &mut out, 1.0);
    // Expected: 1*2.0 + (-1)*3.0 + 0*10.0 + 1*5.0 = 4.0
    assert!((out[0] - 4.0).abs() < 1e-5);

    // Umbral falling factorial: (x)_3 = x(x-1)(x-2)
    assert_eq!(10i32.falling_factorial(3), 10 * 9 * 8);
    assert_eq!(10i16.falling_factorial(3), 10 * 9 * 8);
    assert_eq!(10i8.falling_factorial(3), 10 * 9 * 8);

    // Rational residue carry
    let mut residue = RationalResidue::new(1024);
    assert_eq!(residue.accumulate_and_carry(512), 0);
    assert_eq!(residue.accumulate_and_carry(600), 1); // 512 + 600 = 1112 -> carry 1, rem 88
    assert_eq!(residue.numerator, 88);

    // In-place discrete umbral step
    let mut state = UmbralIntegerState::new(4, 1024);
    let mut coords = [10, -20, 30, -40];
    let diffs = [2, -4, 6, -8];
    state.step_discrete_in_place(&mut coords, &diffs, 1, 1);
    assert_eq!(state.lattice_coords, coords);
}

// ============================================================================
// 7. Candle DType Precision Guards Calibration
// ============================================================================

#[test]
fn test_candle_dtype_precision_guards() {
    let dtypes = [
        (DType::F64, 1e-12, 1e-12),
        (DType::F32, 1e-8, 1e-6),
        (DType::BF16, 1e-6, 1e-4),
        (DType::F16, 1e-4, 1e-3),
        (DType::U8, 0.05, 0.02),
        (DType::U32, 1e-8, 1e-6),
        (DType::I64, 1e-12, 1e-12),
    ];

    for (dt, expected_min_eps, expected_min_guard) in dtypes {
        let guards = SelfOptimizingMetaConfig::precision_guards(dt, 1e-12);
        assert!(
            guards.eff_eps >= expected_min_eps,
            "DType {:?} eff_eps too small: got {:.2e}, expected at least {:.2e}",
            dt,
            guards.eff_eps,
            expected_min_eps
        );
        assert!(
            guards.num_guard >= expected_min_guard,
            "DType {:?} num_guard too small: got {:.2e}, expected at least {:.2e}",
            dt,
            guards.num_guard,
            expected_min_guard
        );
        assert!(guards.min_step_scale > 0.0);
    }
}

// ============================================================================
// 8. Integrator Stability Across Every Numerical Type
// ============================================================================

#[test]
fn test_integrator_stability_across_all_12_numerical_types() {
    let all_types = [
        NumericType::F64,
        NumericType::F32,
        NumericType::Tf32,
        NumericType::F16,
        NumericType::Bf16,
        NumericType::Fp8E4M3,
        NumericType::Fp8E5M2,
        NumericType::Posit32,
        NumericType::Posit16,
        NumericType::Posit8,
        NumericType::Log16,
        NumericType::Log8,
    ];

    for &nt in &all_types {
        // 1. SelfOptimizingMetaIntegrator
        let meta = SelfOptimizingMetaIntegrator::default().with_numeric_type(nt);
        let x0 = vec![1.0, 0.5];
        let (final_x, diag, nfe) = meta.solve_auto(&x0, |x, _t| vec![-x[1], x[0]]);

        assert_eq!(final_x.len(), 2);
        assert!(!diag.is_empty());
        assert!(nfe > 0);
        assert!(
            final_x[0].is_finite() && final_x[1].is_finite(),
            "SelfOptimizingMetaIntegrator diverged for {:?}",
            nt
        );

        // 2. SymplecticPreservingIntegrator
        let symp = SymplecticPreservingIntegrator::default().with_numeric_type(nt);
        let q = vec![1.0, 0.0];
        let p = vec![0.0, 1.0];
        let h = 0.01;
        let target_e = 1.0;
        let (q_next, p_next, e_final) = symp.step(
            &q,
            &p,
            h,
            target_e,
            |q| vec![-q[0], -q[1]],
            |q, p| 0.5 * (p[0] * p[0] + p[1] * p[1] + q[0] * q[0] + q[1] * q[1]),
        );

        assert!(q_next[0].is_finite() && q_next[1].is_finite());
        assert!(p_next[0].is_finite() && p_next[1].is_finite());
        assert!(
            e_final.is_finite(),
            "Symplectic integrator produced non-finite energy for {:?}",
            nt
        );
        // Energy drift must remain reasonably bounded (< 0.25 even for 8-bit types)
        assert!(
            (e_final - target_e).abs() < 0.25,
            "Symplectic integrator energy drift excessive for {:?}: e_final={}",
            nt,
            e_final
        );
    }
}
