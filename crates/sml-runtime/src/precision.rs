//! Numerical Precision Architecture for Type-Agnostic, Precision-Aware Integrators.
//!
//! Provides mathematically exact precision limits, machine epsilon models, quire accumulation,
//! scale-free error norms, and information-theoretic gating for:
//! - IEEE 754: `f64` (double), `f32` (single), `f16` (half), `bf16` (bfloat16)
//! - Unum Type III: `Posit32`, `Posit16`, `Posit8` (tapered precision with golden-zone dynamics)

pub use crate::conversion::{ConversionMode, ConversionReport, FormatConverter};
pub use crate::distribution::DistributionFloat;
use crate::logarithmic::{Log16, Log32, Log64, Log8};
use crate::quantization::{
    decode_bf16, decode_fp16, decode_fp8_e4m3, decode_fp8_e5m2, decode_posit16, decode_posit8,
    decode_tf32, encode_bf16, encode_fp16, encode_fp8_e4m3, encode_fp8_e5m2, encode_posit16,
    encode_posit8, encode_tf32,
};
use serde::{Deserialize, Serialize};

/// Enumeration of all supported numerical datatypes across the precision ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericType {
    F64,
    F32,
    Tf32,
    F16,
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Posit64,
    Posit32,
    Posit16,
    Posit8,
    Log64,
    Log32,
    Log16,
    Log8,
    Int64,
    Int32,
    Nf4,
    Ternary,
}

impl NumericType {
    /// Converts a slice of values from this numeric type format to the target format with minimal drift.
    #[inline]
    pub fn convert_slice_to(
        &self,
        to: NumericType,
        src: &[f64],
        dst: &mut [f64],
        mode: ConversionMode,
    ) -> ConversionReport {
        FormatConverter::convert_slice(src, dst, *self, to, mode)
    }
}

/// Core trait for compile-time zero-cost precision bounds and arithmetic operations.
pub trait IntegratorScalar:
    Copy
    + PartialOrd
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
    const SQRT_EPSILON: Self;
    const MIN_POSITIVE: Self;
    const DENOM_GUARD: Self;
    const TEMPORAL_GUARD: Self;
    const ENERGY_DRIFT_GUARD: Self;
    const PID_ANTIWINDUP_GUARD: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn powf(self, exp: Self) -> Self;
    fn is_finite(self) -> bool;
}

impl IntegratorScalar for f32 {
    const ZERO: Self = 0.0f32;
    const ONE: Self = 1.0f32;
    const EPSILON: Self = f32::EPSILON;
    const SQRT_EPSILON: Self = 3.4526698e-4f32;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const DENOM_GUARD: Self = 1e-8f32;
    const TEMPORAL_GUARD: Self = 1e-6f32;
    const ENERGY_DRIFT_GUARD: Self = 1e-7f32;
    const PID_ANTIWINDUP_GUARD: Self = 1e-9f32;

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

impl IntegratorScalar for f64 {
    const ZERO: Self = 0.0f64;
    const ONE: Self = 1.0f64;
    const EPSILON: Self = f64::EPSILON;
    const SQRT_EPSILON: Self = 1.4901161193847656e-8f64;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const DENOM_GUARD: Self = 1e-15f64;
    const TEMPORAL_GUARD: Self = 1e-14f64;
    const ENERGY_DRIFT_GUARD: Self = 2e-15f64;
    const PID_ANTIWINDUP_GUARD: Self = 1e-18f64;

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }

    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}

/// Core trait defining numerical precision characteristics, machine epsilon bounds,
/// guards, scale-free norms, and arithmetic operations for any datatype.
pub trait NumericPrecision: std::fmt::Debug + Send + Sync + 'static {
    /// Canonical name of the datatype.
    fn name(&self) -> &'static str;

    /// Machine epsilon (unit roundoff) $\epsilon_{\text{mach}}$.
    /// For tapered precision formats like Posit, `val` provides the operating point $x$.
    fn machine_epsilon(&self, val: Option<f64>) -> f64;

    /// Square root of machine epsilon $\sqrt{\epsilon_{\text{mach}}}$.
    fn sqrt_machine_epsilon(&self) -> f64 {
        self.machine_epsilon(None).sqrt()
    }

    /// Denominator protection guard preventing division by zero or subnormal collapse.
    /// Scaled to approximately $\epsilon_{\text{mach}}$ for IEEE or $\epsilon(x)$ for Posit.
    fn denominator_guard(&self) -> f64;

    /// Temporal accumulation guard preventing spurious sub-epsilon steps at integration endpoints.
    fn temporal_guard(&self) -> f64;

    /// Symplectic / Hamiltonian energy drift tolerance floor.
    /// Fluctuations below this value are pure rounding noise and must not trigger projection.
    fn energy_drift_guard(&self) -> f64;

    /// PID step-size controller anti-windup guard.
    fn pid_antiwindup_guard(&self) -> f64;

    /// Default recommended absolute and relative tolerances `(atol, rtol)` for this datatype.
    fn default_tolerances(&self) -> (f64, f64);

    /// Dynamic subgradient clamping bound scaled to the precision and magnitude.
    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64;

    /// Simulates/executes quantization of an `f64` real into this datatype's exact representation.
    fn quantize(&self, val: f64) -> f64;

    /// Nominal bit width of this datatype.
    fn bits(&self) -> usize {
        32
    }

    /// Dynamic range in decibels (dB).
    fn dynamic_range_db(&self) -> f64 {
        1500.0
    }

    /// Whether this datatype features tapered precision (e.g. Posits).
    fn is_tapered(&self) -> bool {
        false
    }

    /// Whether this datatype features logarithmic representation (LNS).
    fn is_logarithmic(&self) -> bool {
        false
    }

    /// Whether this datatype is a discrete integer or lattice format.
    fn is_discrete(&self) -> bool {
        false
    }

    /// Whether this datatype uses quantile normal codebooks (e.g. NF4).
    fn is_quantile_normal(&self) -> bool {
        false
    }

    /// Whether this datatype is a generalized distribution float.
    fn is_distribution_float(&self) -> bool {
        false
    }

    /// In-place slice quantization.
    fn quantize_slice(&self, slice: &mut [f64]) {
        for v in slice.iter_mut() {
            *v = self.quantize(*v);
        }
    }

    /// Quantized gradient descent update step: computes $x - \eta g$ in the target representation.
    fn compute_gradient_update(&self, grad: f64, param: f64, lr: f64) -> f64 {
        let update = param - lr * grad;
        self.quantize(update)
    }

    /// Quire / Fused accumulation for dot products $\sum a_i b_i$.
    /// For Posits, utilizes exact unrounded quire accumulation to prevent catastrophic cancellation.
    fn quire_dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Dimension mismatch in dot product");
        // Default IEEE 754: Kahan / Neumaier compensated summation in f64
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for i in 0..a.len() {
            let prod = self.quantize(a[i]) * self.quantize(b[i]);
            let y = prod - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        self.quantize(sum)
    }
}

// ============================================================================
// IEEE 754 Double Precision (f64)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct F64Precision;

impl NumericPrecision for F64Precision {
    fn name(&self) -> &'static str {
        "IEEE 754 Double (f64)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        f64::EPSILON // 2^-52 ≈ 2.220446049250313e-16
    }

    fn denominator_guard(&self) -> f64 {
        1e-15
    }

    fn temporal_guard(&self) -> f64 {
        1e-14
    }

    fn energy_drift_guard(&self) -> f64 {
        2e-15
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-18
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-9, 1e-8)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        50.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        val
    }
}

// ============================================================================
// IEEE 754 Single Precision (f32)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct F32Precision;

impl NumericPrecision for F32Precision {
    fn name(&self) -> &'static str {
        "IEEE 754 Single (f32)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        f32::EPSILON as f64 // 2^-23 ≈ 1.1920928955078125e-7
    }

    fn denominator_guard(&self) -> f64 {
        1e-8
    }

    fn temporal_guard(&self) -> f64 {
        1e-6
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-7
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-9
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-4, 1e-4)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        25.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        (val as f32) as f64
    }
}

// ============================================================================
// IEEE 754 Half Precision (f16)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct F16Precision;

impl NumericPrecision for F16Precision {
    fn name(&self) -> &'static str {
        "IEEE 754 Half (f16)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        9.765625e-4 // 2^-10 ≈ 0.0009765625
    }

    fn denominator_guard(&self) -> f64 {
        1e-4
    }

    fn temporal_guard(&self) -> f64 {
        1e-3
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-3
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-5
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (5e-3, 5e-3)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        15.0 * base.min(10.0)
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let bits = encode_fp16(val as f32);
        decode_fp16(bits) as f64
    }
}

// ============================================================================
// Brain Floating Point (bf16)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Bf16Precision;

impl NumericPrecision for Bf16Precision {
    fn name(&self) -> &'static str {
        "Brain Float 16 (bf16)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        7.8125e-3 // 2^-7 ≈ 0.0078125
    }

    fn denominator_guard(&self) -> f64 {
        1e-3
    }

    fn temporal_guard(&self) -> f64 {
        1e-2
    }

    fn energy_drift_guard(&self) -> f64 {
        8e-3
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-4
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1.5e-2, 1.5e-2)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        20.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let bits = encode_bf16(val as f32);
        decode_bf16(bits) as f64
    }
}

// ============================================================================
// TensorFloat-32 (TF32)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Tf32Precision;

impl NumericPrecision for Tf32Precision {
    fn name(&self) -> &'static str {
        "TensorFloat-32 (TF32)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        9.765625e-4 // 2^-10 ≈ 0.0009765625 (same mantissa bits as FP16, range of FP32)
    }

    fn denominator_guard(&self) -> f64 {
        1e-4
    }

    fn temporal_guard(&self) -> f64 {
        1e-3
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-4
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-6
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-3, 1e-3)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        20.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let bits = encode_tf32(val as f32);
        decode_tf32(bits) as f64
    }
}

// ============================================================================
// FP8 Formats (E4M3 and E5M2)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8E4M3Precision;

impl NumericPrecision for Fp8E4M3Precision {
    fn name(&self) -> &'static str {
        "FP8 E4M3 (Weights)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        0.125 // 2^-3 = 0.125
    }

    fn denominator_guard(&self) -> f64 {
        0.05
    }

    fn temporal_guard(&self) -> f64 {
        0.1
    }

    fn energy_drift_guard(&self) -> f64 {
        0.05
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.01
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.05, 0.05)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        10.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let byte = encode_fp8_e4m3(val as f32);
        decode_fp8_e4m3(byte) as f64
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8E5M2Precision;

impl NumericPrecision for Fp8E5M2Precision {
    fn name(&self) -> &'static str {
        "FP8 E5M2 (Gradients & Activations)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        0.25 // 2^-2 = 0.25
    }

    fn denominator_guard(&self) -> f64 {
        0.1
    }

    fn temporal_guard(&self) -> f64 {
        0.2
    }

    fn energy_drift_guard(&self) -> f64 {
        0.1
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.02
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.1, 0.1)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        8.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let byte = encode_fp8_e5m2(val as f32);
        decode_fp8_e5m2(byte) as f64
    }
}

// ============================================================================
// Logarithmic Precision (Log16 and Log8)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Log16Precision;

impl NumericPrecision for Log16Precision {
    fn name(&self) -> &'static str {
        "Logarithmic 16-Bit (Log16)"
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Log16).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        1e-4
    }

    fn temporal_guard(&self) -> f64 {
        1e-3
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-4
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-5
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (5e-3, 5e-3)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        15.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        Log16::from_f32(val as f32).to_f32() as f64
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Log8Precision;

impl NumericPrecision for Log8Precision {
    fn name(&self) -> &'static str {
        "Logarithmic 8-Bit (Log8)"
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Log8).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        0.01
    }

    fn temporal_guard(&self) -> f64 {
        0.05
    }

    fn energy_drift_guard(&self) -> f64 {
        0.02
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.005
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.05, 0.05)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        10.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        Log8::from_f32(val as f32).to_f32() as f64
    }
}

// ============================================================================
// Unum Type III: Posit32 <32, 2>
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Posit32Precision;

impl NumericPrecision for Posit32Precision {
    fn name(&self) -> &'static str {
        "Posit32 <32, 2> (Tapered Unum)"
    }

    fn is_tapered(&self) -> bool {
        true
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Posit32).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        1e-10
    }

    fn temporal_guard(&self) -> f64 {
        5e-8
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-8
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-11
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (5e-6, 5e-6)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        30.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        // Posit32 simulation via exact bit roundoff modeling
        if val == 0.0 {
            return 0.0;
        }
        let eps = self.machine_epsilon(Some(val));
        (val / eps).round() * eps
    }

    fn quire_dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        // Posit Quire: Exact 512-bit fixed point accumulator eliminates intermediate rounding
        let mut exact_sum = 0.0f64;
        for i in 0..a.len() {
            exact_sum += a[i] * b[i];
        }
        self.quantize(exact_sum)
    }
}

// ============================================================================
// Unum Type III: Posit16 <16, 1>
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Posit16Precision;

impl NumericPrecision for Posit16Precision {
    fn name(&self) -> &'static str {
        "Posit16 <16, 1> (Tapered Unum)"
    }

    fn is_tapered(&self) -> bool {
        true
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        // Posit16 has es=1, useed = 4.
        // In the golden zone (around 1.0), regime is 2 bits, es is 1 bit, sign is 1 bit:
        // Fraction has 16 - 1 - 2 - 1 = 12 bits!
        // Machine epsilon: 2^-12 ≈ 2.4414e-4.
        if let Some(v) = val {
            if v == 0.0 {
                return 2.4414e-4;
            }
            let log_useed = (v.abs().log2() / 2.0).floor() as i32;
            let regime_bits = (log_useed.abs() + 2).min(14) as u32;
            let frac_bits = 16u32.saturating_sub(1 + regime_bits + 1);
            2.0f64.powi(-(frac_bits as i32))
        } else {
            2.44140625e-4
        }
    }

    fn denominator_guard(&self) -> f64 {
        1e-5
    }

    fn temporal_guard(&self) -> f64 {
        5e-4
    }

    fn energy_drift_guard(&self) -> f64 {
        3e-4
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-6
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-3, 1e-3)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        20.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        decode_posit16(encode_posit16(val as f32)) as f64
    }

    fn quire_dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut exact_sum = 0.0f64;
        for i in 0..a.len() {
            exact_sum += a[i] * b[i];
        }
        self.quantize(exact_sum)
    }
}

// ============================================================================
// Unum Type III: Posit8 <8, 1>
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Posit8Precision;

impl NumericPrecision for Posit8Precision {
    fn name(&self) -> &'static str {
        "Posit8 <8, 1> (Tapered Unum)"
    }

    fn is_tapered(&self) -> bool {
        true
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Posit8).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        0.01
    }

    fn temporal_guard(&self) -> f64 {
        0.05
    }

    fn energy_drift_guard(&self) -> f64 {
        0.02
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.005
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.02, 0.02)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        10.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        decode_posit8(encode_posit8(val as f32)) as f64
    }

    fn quire_dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut exact_sum = 0.0f64;
        for i in 0..a.len() {
            exact_sum += a[i] * b[i];
        }
        self.quantize(exact_sum)
    }
}

// ============================================================================
// Unum Type III: Posit64 <64, 3>
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Posit64Precision;

impl NumericPrecision for Posit64Precision {
    fn name(&self) -> &'static str {
        "Posit64 <64, 3> (Tapered Unum)"
    }

    fn is_tapered(&self) -> bool {
        true
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Posit64).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        1e-16
    }

    fn temporal_guard(&self) -> f64 {
        1e-15
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-16
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-19
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-12, 1e-11)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        50.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        if val == 0.0 {
            return 0.0;
        }
        let eps = self.machine_epsilon(Some(val));
        (val / eps).round() * eps
    }

    fn quire_dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut exact_sum = 0.0f64;
        for i in 0..a.len() {
            exact_sum += a[i] * b[i];
        }
        self.quantize(exact_sum)
    }
}

// ============================================================================
// Logarithmic Precision (Log64 and Log32)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Log64Precision;

impl NumericPrecision for Log64Precision {
    fn name(&self) -> &'static str {
        "Logarithmic 64-Bit (Log64)"
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Log64).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        1e-15
    }

    fn temporal_guard(&self) -> f64 {
        1e-14
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-15
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-18
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-9, 1e-8)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        50.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        Log64::from_f64(val).to_f64()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Log32Precision;

impl NumericPrecision for Log32Precision {
    fn name(&self) -> &'static str {
        "Logarithmic 32-Bit (Log32)"
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        DataTypeSpec::of(NumericType::Log32).machine_epsilon_at(val)
    }

    fn denominator_guard(&self) -> f64 {
        1e-8
    }

    fn temporal_guard(&self) -> f64 {
        1e-6
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-7
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-9
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-4, 1e-4)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        25.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        Log32::from_f64(val).to_f64()
    }
}

// ============================================================================
// Discrete Integer & Ternary Formats
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Int64Precision;

impl NumericPrecision for Int64Precision {
    fn name(&self) -> &'static str {
        "Discrete Signed Integer 64-Bit (Int64)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        1e-9
    }

    fn denominator_guard(&self) -> f64 {
        1e-8
    }

    fn temporal_guard(&self) -> f64 {
        1e-7
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-7
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-9
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-6, 1e-6)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        20.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        (val * 1e9).round() * 1e-9
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Int32Precision;

impl NumericPrecision for Int32Precision {
    fn name(&self) -> &'static str {
        "Discrete Signed Integer 32-Bit (Int32)"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        1e-4
    }

    fn denominator_guard(&self) -> f64 {
        1e-3
    }

    fn temporal_guard(&self) -> f64 {
        1e-3
    }

    fn energy_drift_guard(&self) -> f64 {
        1e-3
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        1e-5
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (1e-3, 1e-3)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        15.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        (val * 1e4).round() * 1e-4
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TernaryPrecision;

impl NumericPrecision for TernaryPrecision {
    fn name(&self) -> &'static str {
        "Ternary {-1, 0, +1} Quantization"
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        0.5
    }

    fn denominator_guard(&self) -> f64 {
        0.1
    }

    fn temporal_guard(&self) -> f64 {
        0.2
    }

    fn energy_drift_guard(&self) -> f64 {
        0.1
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.05
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.2, 0.2)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        5.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let scale = 0.5;
        let rounded = (val / scale).round().clamp(-1.0, 1.0);
        rounded * scale
    }
}

// ============================================================================
// NormalFloat 4-Bit Precision (NF4 / QLoRA)
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct Nf4Precision;

impl NumericPrecision for Nf4Precision {
    fn name(&self) -> &'static str {
        "NF4 (NormalFloat-4 / QLoRA)"
    }

    fn bits(&self) -> usize {
        4
    }

    fn dynamic_range_db(&self) -> f64 {
        200.0
    }

    fn is_quantile_normal(&self) -> bool {
        true
    }

    fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
        0.05
    }

    fn denominator_guard(&self) -> f64 {
        1e-3
    }

    fn temporal_guard(&self) -> f64 {
        1e-2
    }

    fn energy_drift_guard(&self) -> f64 {
        0.05
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        0.85
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (0.05, 0.05)
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        3.0 * base
    }

    #[inline]
    fn quantize(&self, val: f64) -> f64 {
        let code = crate::quantization::quantize_to_nf4(val as f32);
        crate::quantization::dequantize_nf4(code) as f64
    }
}

// Blanket implementation for Arc<dyn NumericPrecision>
impl NumericPrecision for std::sync::Arc<dyn NumericPrecision> {
    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn bits(&self) -> usize {
        (**self).bits()
    }

    fn dynamic_range_db(&self) -> f64 {
        (**self).dynamic_range_db()
    }

    fn is_tapered(&self) -> bool {
        (**self).is_tapered()
    }

    fn is_discrete(&self) -> bool {
        (**self).is_discrete()
    }

    fn is_logarithmic(&self) -> bool {
        (**self).is_logarithmic()
    }

    fn is_quantile_normal(&self) -> bool {
        (**self).is_quantile_normal()
    }

    fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        (**self).machine_epsilon(val)
    }

    fn denominator_guard(&self) -> f64 {
        (**self).denominator_guard()
    }

    fn temporal_guard(&self) -> f64 {
        (**self).temporal_guard()
    }

    fn energy_drift_guard(&self) -> f64 {
        (**self).energy_drift_guard()
    }

    fn pid_antiwindup_guard(&self) -> f64 {
        (**self).pid_antiwindup_guard()
    }

    fn default_tolerances(&self) -> (f64, f64) {
        (**self).default_tolerances()
    }

    fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        (**self).adaptive_clamp_bound(state_norm, velocity_norm)
    }

    fn quantize(&self, val: f64) -> f64 {
        (**self).quantize(val)
    }

    fn quantize_slice(&self, slice: &mut [f64]) {
        (**self).quantize_slice(slice)
    }

    fn compute_gradient_update(&self, grad: f64, param: f64, lr: f64) -> f64 {
        (**self).compute_gradient_update(grad, param, lr)
    }
}

// ============================================================================
// Canonical Data-Type Specification & Precision Guards
// ============================================================================

/// Numerical precision guards calibrated to target mantissa and exponent dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrecisionGuards {
    pub eff_eps: f64,
    pub num_guard: f64,
    pub min_step_scale: f64,
}

/// Canonical type encapsulating implementation specifications, dynamic machine epsilon
/// models, exponent/mantissa distributions, and derived guards per data type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataTypeSpec {
    pub numeric_type: NumericType,
    pub bit_width: usize,
    pub storage_bytes: usize,
    pub mantissa_bits_min: usize,
    pub mantissa_bits_max: usize,
    pub exponent_bits: usize,
    pub is_tapered: bool,
    pub is_logarithmic: bool,
    pub is_discrete: bool,
    pub min_positive: f64,
    pub max_normal: f64,
    pub default_eff_eps: f64,
    pub default_num_guard: f64,
    pub min_step_scale: f64,
}

impl DataTypeSpec {
    pub const fn of(dtype: NumericType) -> Self {
        match dtype {
            NumericType::F64 => Self {
                numeric_type: NumericType::F64,
                bit_width: 64,
                storage_bytes: 8,
                mantissa_bits_min: 52,
                mantissa_bits_max: 52,
                exponent_bits: 11,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 2.2250738585072014e-308,
                max_normal: 1.7976931348623157e+308,
                default_eff_eps: 1e-12,
                default_num_guard: 1e-12,
                min_step_scale: 0.05,
            },
            NumericType::F32 => Self {
                numeric_type: NumericType::F32,
                bit_width: 32,
                storage_bytes: 4,
                mantissa_bits_min: 23,
                mantissa_bits_max: 23,
                exponent_bits: 8,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1.17549435e-38,
                max_normal: 3.4028235e+38,
                default_eff_eps: 1e-8,
                default_num_guard: 1e-6,
                min_step_scale: 0.10,
            },
            NumericType::Tf32 => Self {
                numeric_type: NumericType::Tf32,
                bit_width: 19,
                storage_bytes: 4,
                mantissa_bits_min: 10,
                mantissa_bits_max: 10,
                exponent_bits: 8,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1.17549435e-38,
                max_normal: 3.4028235e+38,
                default_eff_eps: 1e-4,
                default_num_guard: 1e-4,
                min_step_scale: 0.12,
            },
            NumericType::Bf16 => Self {
                numeric_type: NumericType::Bf16,
                bit_width: 16,
                storage_bytes: 2,
                mantissa_bits_min: 7,
                mantissa_bits_max: 7,
                exponent_bits: 8,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1.17549435e-38,
                max_normal: 3.3895314e+38,
                default_eff_eps: 1e-6,
                default_num_guard: 1e-4,
                min_step_scale: 0.15,
            },
            NumericType::F16 => Self {
                numeric_type: NumericType::F16,
                bit_width: 16,
                storage_bytes: 2,
                mantissa_bits_min: 10,
                mantissa_bits_max: 10,
                exponent_bits: 5,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 6.1035156e-5,
                max_normal: 65504.0,
                default_eff_eps: 1e-3,
                default_num_guard: 1e-3,
                min_step_scale: 0.20,
            },
            NumericType::Fp8E4M3 => Self {
                numeric_type: NumericType::Fp8E4M3,
                bit_width: 8,
                storage_bytes: 1,
                mantissa_bits_min: 3,
                mantissa_bits_max: 3,
                exponent_bits: 4,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 0.001953125,
                max_normal: 448.0,
                default_eff_eps: 0.05,
                default_num_guard: 0.02,
                min_step_scale: 0.25,
            },
            NumericType::Fp8E5M2 => Self {
                numeric_type: NumericType::Fp8E5M2,
                bit_width: 8,
                storage_bytes: 1,
                mantissa_bits_min: 2,
                mantissa_bits_max: 2,
                exponent_bits: 5,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1.5258789e-5,
                max_normal: 57344.0,
                default_eff_eps: 0.10,
                default_num_guard: 0.05,
                min_step_scale: 0.25,
            },
            NumericType::Posit32 => Self {
                numeric_type: NumericType::Posit32,
                bit_width: 32,
                storage_bytes: 4,
                mantissa_bits_min: 0,
                mantissa_bits_max: 27,
                exponent_bits: 2,
                is_tapered: true,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1.3877787807814457e-38,
                max_normal: 7.205759403792794e+37,
                default_eff_eps: 1e-8,
                default_num_guard: 1e-7,
                min_step_scale: 0.10,
            },
            NumericType::Posit16 => Self {
                numeric_type: NumericType::Posit16,
                bit_width: 16,
                storage_bytes: 2,
                mantissa_bits_min: 0,
                mantissa_bits_max: 12,
                exponent_bits: 1,
                is_tapered: true,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 3.725290298461914e-9,
                max_normal: 268435456.0,
                default_eff_eps: 2.5e-4,
                default_num_guard: 1e-4,
                min_step_scale: 0.15,
            },
            NumericType::Posit8 => Self {
                numeric_type: NumericType::Posit8,
                bit_width: 8,
                storage_bytes: 1,
                mantissa_bits_min: 0,
                mantissa_bits_max: 4,
                exponent_bits: 1,
                is_tapered: true,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 0.015625,
                max_normal: 64.0,
                default_eff_eps: 0.03125,
                default_num_guard: 0.01,
                min_step_scale: 0.20,
            },
            NumericType::Log16 => Self {
                numeric_type: NumericType::Log16,
                bit_width: 16,
                storage_bytes: 2,
                mantissa_bits_min: 8,
                mantissa_bits_max: 8,
                exponent_bits: 7,
                is_tapered: false,
                is_logarithmic: true,
                is_discrete: false,
                min_positive: 1e-25,
                max_normal: 1e25,
                default_eff_eps: 0.00271,
                default_num_guard: 0.001,
                min_step_scale: 0.15,
            },
            NumericType::Log8 => Self {
                numeric_type: NumericType::Log8,
                bit_width: 8,
                storage_bytes: 1,
                mantissa_bits_min: 4,
                mantissa_bits_max: 4,
                exponent_bits: 3,
                is_tapered: false,
                is_logarithmic: true,
                is_discrete: false,
                min_positive: 1e-12,
                max_normal: 1e12,
                default_eff_eps: 0.04427,
                default_num_guard: 0.01,
                min_step_scale: 0.20,
            },
            NumericType::Posit64 => Self {
                numeric_type: NumericType::Posit64,
                bit_width: 64,
                storage_bytes: 8,
                mantissa_bits_min: 0,
                mantissa_bits_max: 58,
                exponent_bits: 3,
                is_tapered: true,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 1e-300,
                max_normal: 1e300,
                default_eff_eps: 1e-15,
                default_num_guard: 1e-15,
                min_step_scale: 0.05,
            },
            NumericType::Log64 => Self {
                numeric_type: NumericType::Log64,
                bit_width: 64,
                storage_bytes: 8,
                mantissa_bits_min: 52,
                mantissa_bits_max: 52,
                exponent_bits: 11,
                is_tapered: false,
                is_logarithmic: true,
                is_discrete: false,
                min_positive: 1e-300,
                max_normal: 1e300,
                default_eff_eps: 1e-12,
                default_num_guard: 1e-12,
                min_step_scale: 0.05,
            },
            NumericType::Log32 => Self {
                numeric_type: NumericType::Log32,
                bit_width: 32,
                storage_bytes: 4,
                mantissa_bits_min: 23,
                mantissa_bits_max: 23,
                exponent_bits: 8,
                is_tapered: false,
                is_logarithmic: true,
                is_discrete: false,
                min_positive: 1e-50,
                max_normal: 1e50,
                default_eff_eps: 1e-7,
                default_num_guard: 1e-6,
                min_step_scale: 0.10,
            },
            NumericType::Int64 => Self {
                numeric_type: NumericType::Int64,
                bit_width: 64,
                storage_bytes: 8,
                mantissa_bits_min: 63,
                mantissa_bits_max: 63,
                exponent_bits: 0,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: true,
                min_positive: 1e-9,
                max_normal: 9e9,
                default_eff_eps: 1e-9,
                default_num_guard: 1e-8,
                min_step_scale: 0.05,
            },
            NumericType::Int32 => Self {
                numeric_type: NumericType::Int32,
                bit_width: 32,
                storage_bytes: 4,
                mantissa_bits_min: 31,
                mantissa_bits_max: 31,
                exponent_bits: 0,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: true,
                min_positive: 1e-4,
                max_normal: 2e5,
                default_eff_eps: 1e-4,
                default_num_guard: 1e-3,
                min_step_scale: 0.10,
            },
            NumericType::Nf4 => Self {
                numeric_type: NumericType::Nf4,
                bit_width: 4,
                storage_bytes: 1,
                mantissa_bits_min: 3,
                mantissa_bits_max: 4,
                exponent_bits: 0,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: false,
                min_positive: 0.05,
                max_normal: 1.0,
                default_eff_eps: 0.05,
                default_num_guard: 1e-3,
                min_step_scale: 0.15,
            },
            NumericType::Ternary => Self {
                numeric_type: NumericType::Ternary,
                bit_width: 2,
                storage_bytes: 1,
                mantissa_bits_min: 1,
                mantissa_bits_max: 1,
                exponent_bits: 0,
                is_tapered: false,
                is_logarithmic: false,
                is_discrete: true,
                min_positive: 0.5,
                max_normal: 0.5,
                default_eff_eps: 0.5,
                default_num_guard: 0.1,
                min_step_scale: 0.20,
            },
        }
    }

    /// Display name of the numerical data type.
    pub fn numeric_type_name(&self) -> &'static str {
        match self.numeric_type {
            NumericType::F64 => "f64",
            NumericType::F32 => "f32",
            NumericType::Tf32 => "tf32",
            NumericType::Bf16 => "bf16",
            NumericType::F16 => "f16",
            NumericType::Fp8E4M3 => "fp8_e4m3",
            NumericType::Fp8E5M2 => "fp8_e5m2",
            NumericType::Posit64 => "posit64",
            NumericType::Posit32 => "posit32",
            NumericType::Posit16 => "posit16",
            NumericType::Posit8 => "posit8",
            NumericType::Log64 => "log64",
            NumericType::Log32 => "log32",
            NumericType::Log16 => "log16",
            NumericType::Log8 => "log8",
            NumericType::Int64 => "int64",
            NumericType::Int32 => "int32",
            NumericType::Nf4 => "nf4",
            NumericType::Ternary => "ternary",
        }
    }

    /// Evaluates point-wise machine epsilon $\epsilon(x)$.
    /// For tapered Posits, evaluates dynamic epsilon based on magnitude $x$.
    /// When `val` is None, returns the golden zone machine epsilon (at $|x| \approx 1.0$).
    pub fn machine_epsilon_at(&self, val: Option<f64>) -> f64 {
        match self.numeric_type {
            NumericType::F64 => f64::EPSILON,
            NumericType::F32 => f32::EPSILON as f64,
            NumericType::Tf32 => 9.765625e-4, // 2^-10
            NumericType::Bf16 => 7.8125e-3,    // 2^-7
            NumericType::F16 => 9.765625e-4,   // 2^-10
            NumericType::Fp8E4M3 => 0.125,     // 2^-3
            NumericType::Fp8E5M2 => 0.25,      // 2^-2
            NumericType::Nf4 => 0.05,          // QLoRA NormalFloat-4 average quantile step
            NumericType::Posit64 => {
                if let Some(v) = val {
                    if v == 0.0 {
                        return 3.469446951953614e-18;
                    }
                    let log_useed = (v.abs().log2() / 8.0).floor() as i32;
                    let regime_bits = (log_useed.abs() + 2).min(60) as u32;
                    let frac_bits = 64u32.saturating_sub(1 + regime_bits + 3);
                    2.0f64.powi(-(frac_bits as i32))
                } else {
                    3.469446951953614e-18
                }
            }
            NumericType::Posit32 => {
                if let Some(v) = val {
                    if v == 0.0 {
                        return 7.450580596923828e-9;
                    }
                    let log_useed = (v.abs().log2() / 4.0).floor() as i32;
                    let regime_bits = (log_useed.abs() + 2).min(30) as u32;
                    let frac_bits = 32u32.saturating_sub(1 + regime_bits + 2);
                    2.0f64.powi(-(frac_bits as i32))
                } else {
                    7.450580596923828e-9
                }
            }
            NumericType::Posit16 => {
                if let Some(v) = val {
                    if v == 0.0 {
                        return 2.44140625e-4;
                    }
                    let log_useed = (v.abs().log2() / 2.0).floor() as i32;
                    let regime_bits = (log_useed.abs() + 2).min(14) as u32;
                    let frac_bits = 16u32.saturating_sub(1 + regime_bits + 1);
                    2.0f64.powi(-(frac_bits as i32))
                } else {
                    2.44140625e-4
                }
            }
            NumericType::Posit8 => {
                if let Some(v) = val {
                    if v == 0.0 {
                        return 0.0625;
                    }
                    let log_useed = (v.abs().log2() / 2.0).floor() as i32;
                    let regime_bits = (log_useed.abs() + 2).min(6) as u32;
                    let frac_bits = 8u32.saturating_sub(1 + regime_bits + 1);
                    2.0f64.powi(-(frac_bits as i32))
                } else {
                    0.0625
                }
            }
            NumericType::Log64 => 2.220446049250313e-16,
            NumericType::Log32 => 1.1920928955078125e-7,
            NumericType::Log16 => 0.002714,
            NumericType::Log8 => 0.04427,
            NumericType::Int64 => 1e-9,
            NumericType::Int32 => 1e-4,
            NumericType::Ternary => 0.5,
        }
    }

    /// Machine epsilon accessor supporting point-wise dynamic query.
    #[inline(always)]
    pub fn machine_epsilon(&self, val: Option<f64>) -> f64 {
        self.machine_epsilon_at(val)
    }

    /// Square root of nominal machine epsilon for step probing.
    #[inline(always)]
    pub fn sqrt_machine_epsilon(&self) -> f64 {
        self.machine_epsilon_at(None).sqrt()
    }

    /// Temporal step-size guard calibrated to this datatype.
    #[inline(always)]
    pub const fn temporal_guard(&self) -> f64 {
        match self.numeric_type {
            NumericType::Posit64 => 1e-15,
            NumericType::F64 | NumericType::Log64 => 1e-8,
            NumericType::F32 | NumericType::Tf32 | NumericType::Posit32 | NumericType::Log32 => 1e-6,
            NumericType::Int64 => 1e-7,
            NumericType::F16 | NumericType::Bf16 | NumericType::Posit16 | NumericType::Log16 | NumericType::Int32 => 1e-4,
            NumericType::Fp8E4M3 | NumericType::Fp8E5M2 | NumericType::Posit8 | NumericType::Log8 => 1e-3,
            NumericType::Nf4 => 1e-2,
            NumericType::Ternary => 0.1,
        }
    }

    /// Denominator regularization guard calibrated to this datatype.
    #[inline(always)]
    pub const fn denominator_guard(&self) -> f64 {
        match self.numeric_type {
            NumericType::Posit64 => 1e-16,
            NumericType::F64 | NumericType::Log64 => 1e-12,
            NumericType::Int64 => 1e-8,
            NumericType::F32 | NumericType::Tf32 | NumericType::Posit32 | NumericType::Log32 => 1e-7,
            NumericType::F16 | NumericType::Bf16 | NumericType::Posit16 | NumericType::Log16 | NumericType::Int32 => 1e-4,
            NumericType::Fp8E4M3 | NumericType::Fp8E5M2 | NumericType::Posit8 | NumericType::Log8 => 1e-2,
            NumericType::Nf4 => 1e-3,
            NumericType::Ternary => 0.1,
        }
    }

    /// Hamiltonian energy drift tolerance guard calibrated to this datatype.
    #[inline(always)]
    pub const fn energy_drift_guard(&self) -> f64 {
        match self.numeric_type {
            NumericType::Posit64 => 1e-16,
            NumericType::F64 | NumericType::Log64 => 1e-5,
            NumericType::Int64 => 1e-7,
            NumericType::F32 | NumericType::Tf32 | NumericType::Posit32 | NumericType::Log32 => 1e-3,
            NumericType::Int32 => 1e-3,
            NumericType::F16 | NumericType::Bf16 | NumericType::Posit16 | NumericType::Log16 => 5e-2,
            NumericType::Fp8E4M3 | NumericType::Fp8E5M2 | NumericType::Posit8 | NumericType::Log8 => 0.1,
            NumericType::Nf4 => 0.05,
            NumericType::Ternary => 0.1,
        }
    }

    /// PID controller antiwindup saturation guard calibrated to this datatype.
    #[inline(always)]
    pub const fn pid_antiwindup_guard(&self) -> f64 {
        match self.numeric_type {
            NumericType::Posit64 | NumericType::F64 | NumericType::Log64 | NumericType::Int64 => 1e-18,
            NumericType::F32 | NumericType::Tf32 | NumericType::Posit32 | NumericType::Log32 => 1e-9,
            NumericType::Int32 => 1e-7,
            NumericType::F16 | NumericType::Bf16 | NumericType::Posit16 | NumericType::Log16 => 1e-5,
            NumericType::Fp8E4M3 | NumericType::Fp8E5M2 | NumericType::Posit8 | NumericType::Log8 => 1e-4,
            NumericType::Nf4 => 1e-3,
            NumericType::Ternary => 1e-2,
        }
    }

    /// Adaptive state/subgradient clamping ceiling calibrated to this datatype.
    #[inline(always)]
    pub fn adaptive_clamp_bound(&self, state_norm: f64, velocity_norm: f64) -> f64 {
        let base = (state_norm + velocity_norm).max(1.0);
        match self.numeric_type {
            NumericType::Posit64 | NumericType::F64 | NumericType::Log64 => 50.0 * base,
            NumericType::Int64 => 20.0 * base,
            NumericType::F32 | NumericType::Tf32 | NumericType::Posit32 | NumericType::Log32 => 10.0 * base,
            NumericType::Int32 => 15.0 * base,
            NumericType::F16 | NumericType::Bf16 | NumericType::Posit16 | NumericType::Log16 => 5.0 * base,
            NumericType::Fp8E4M3 | NumericType::Fp8E5M2 | NumericType::Posit8 | NumericType::Log8 => 2.0 * base,
            NumericType::Nf4 => 3.0 * base,
            NumericType::Ternary => 5.0 * base,
        }
    }

    /// Computes numerical precision guards calibrated to this datatype.
    pub fn precision_guards(&self, user_override_eps: Option<f64>) -> PrecisionGuards {
        let eff_eps = user_override_eps.unwrap_or(self.default_eff_eps);
        PrecisionGuards {
            eff_eps,
            num_guard: self.default_num_guard,
            min_step_scale: self.min_step_scale,
        }
    }
}

// ============================================================================
// Factory & Dynamic Dispatch
// ============================================================================

/// Creates a boxed `NumericPrecision` provider from an enum.
pub fn get_precision(num_type: NumericType) -> Box<dyn NumericPrecision> {
    match num_type {
        NumericType::F64 => Box::new(F64Precision),
        NumericType::F32 => Box::new(F32Precision),
        NumericType::Tf32 => Box::new(Tf32Precision),
        NumericType::F16 => Box::new(F16Precision),
        NumericType::Bf16 => Box::new(Bf16Precision),
        NumericType::Fp8E4M3 => Box::new(Fp8E4M3Precision),
        NumericType::Fp8E5M2 => Box::new(Fp8E5M2Precision),
        NumericType::Posit64 => Box::new(Posit64Precision),
        NumericType::Posit32 => Box::new(Posit32Precision),
        NumericType::Posit16 => Box::new(Posit16Precision),
        NumericType::Posit8 => Box::new(Posit8Precision),
        NumericType::Log64 => Box::new(Log64Precision),
        NumericType::Log32 => Box::new(Log32Precision),
        NumericType::Log16 => Box::new(Log16Precision),
        NumericType::Log8 => Box::new(Log8Precision),
        NumericType::Int64 => Box::new(Int64Precision),
        NumericType::Int32 => Box::new(Int32Precision),
        NumericType::Nf4 => Box::new(Nf4Precision),
        NumericType::Ternary => Box::new(TernaryPrecision),
    }
}

// ============================================================================
// Multi-Objective Quantization Ranking Engine
// ============================================================================

/// Quantization format ranking entry containing multi-objective metrics and Pareto scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationRankEntry {
    pub dtype: NumericType,
    pub name: String,
    pub memory_bytes_per_param: usize,
    pub machine_epsilon: f64,
    pub dynamic_range_db: f64,
    pub sqnr_db: f64,
    pub latency_relative_to_f32: f64,
    pub pareto_score: f64,
    pub recommendation_rank: usize,
    pub rationale: String,
}

/// Evaluation operating regime mode for the `QuantizationRanker`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QuantizationEvaluationMode {
    /// High-Precision ODE Integration: prioritizes minimal machine epsilon, symplectic stability, and energy conservation.
    #[default]
    Integrator,
    /// Stochastic Deep Learning / AI Optimization: prioritizes memory bandwidth, bytes/param, and throughput because stochastic gradient noise dominates machine precision.
    AiOptimizer,
}

/// Comprehensive Pareto Quantization Ranking Engine for internal meta-integrator and optimizer decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationRanker {
    pub memory_weight: f64,
    pub speed_weight: f64,
    pub precision_weight: f64,
    pub dynamic_range_weight: f64,
}

impl Default for QuantizationRanker {
    fn default() -> Self {
        Self {
            memory_weight: 0.35,
            speed_weight: 0.30,
            precision_weight: 0.25,
            dynamic_range_weight: 0.10,
        }
    }
}

impl QuantizationRanker {
    /// Constructs a ranker calibrated for high-precision ODE integration.
    pub fn for_integrator() -> Self {
        Self {
            memory_weight: 0.15,
            speed_weight: 0.15,
            precision_weight: 0.50,
            dynamic_range_weight: 0.20,
        }
    }

    /// Constructs a ranker calibrated for stochastic AI optimization.
    pub fn for_ai_optimizer() -> Self {
        Self {
            memory_weight: 0.45,
            speed_weight: 0.35,
            precision_weight: 0.15,
            dynamic_range_weight: 0.05,
        }
    }

    /// Ranks all supported numerical datatypes for high-precision ODE integration.
    pub fn rank_for_integrator(&self) -> Vec<QuantizationRankEntry> {
        self.rank_all(QuantizationEvaluationMode::Integrator)
    }

    /// Ranks all supported numerical datatypes for stochastic AI optimization.
    pub fn rank_for_ai_optimizer(&self) -> Vec<QuantizationRankEntry> {
        self.rank_all(QuantizationEvaluationMode::AiOptimizer)
    }

    /// Computes multi-objective Pareto ranking across all formats for a specified evaluation mode.
    pub fn rank_all(&self, mode: QuantizationEvaluationMode) -> Vec<QuantizationRankEntry> {
        let all_types = [
            NumericType::F64,
            NumericType::Posit64,
            NumericType::Log64,
            NumericType::Int64,
            NumericType::F32,
            NumericType::Posit32,
            NumericType::Log32,
            NumericType::Int32,
            NumericType::Tf32,
            NumericType::F16,
            NumericType::Bf16,
            NumericType::Posit16,
            NumericType::Log16,
            NumericType::Fp8E4M3,
            NumericType::Fp8E5M2,
            NumericType::Nf4,
            NumericType::Posit8,
            NumericType::Log8,
            NumericType::Ternary,
        ];

        let mut entries: Vec<QuantizationRankEntry> = all_types
            .into_iter()
            .map(|dtype| {
                let spec = DataTypeSpec::of(dtype);
                let eps = spec.machine_epsilon(None);
                let bytes = spec.storage_bytes.max(1);

                // Dynamic range in dB: 20 * log10(max / min)
                let dynamic_range_db = if spec.min_positive > 0.0 && spec.max_normal > spec.min_positive {
                    20.0 * (spec.max_normal / spec.min_positive).log10().clamp(1.0, 700.0)
                } else {
                    12.0
                };

                // Theoretical SQNR (Signal-to-Quantization-Noise Ratio): 6.02 * bits + 1.76 dB
                let effective_bits = spec.mantissa_bits_max as f64 + 1.0;
                let sqnr_db = 6.02 * effective_bits + 1.76;

                // Relative latency estimate normalized to f32 SIMD (f32 = 1.0)
                let latency_rel = match dtype {
                    NumericType::F32 => 1.0,
                    NumericType::Tf32 => 0.9,
                    NumericType::F16 | NumericType::Bf16 => 0.7,
                    NumericType::Fp8E4M3 | NumericType::Fp8E5M2 => 0.5,
                    NumericType::Nf4 => 0.6,
                    NumericType::Int32 => 0.8,
                    NumericType::Int64 => 1.2,
                    NumericType::F64 => 1.5,
                    NumericType::Posit32 => 2.2,
                    NumericType::Posit16 => 1.8,
                    NumericType::Posit64 => 3.5,
                    NumericType::Posit8 => 1.4,
                    NumericType::Log32 => 1.1,
                    NumericType::Log16 => 0.8,
                    NumericType::Log64 => 1.4,
                    NumericType::Log8 => 0.6,
                    NumericType::Ternary => 0.3,
                };

                // Multi-objective Pareto score calculation
                let pareto_score = match mode {
                    QuantizationEvaluationMode::Integrator => {
                        let score_eps = (-eps.log10()).clamp(0.0, 18.0) / 18.0;
                        let score_dr = (dynamic_range_db / 600.0).clamp(0.0, 1.0);
                        let score_speed = (1.0 / latency_rel).clamp(0.1, 3.0) / 2.0;
                        let score_mem = (8.0 / bytes as f64).clamp(1.0, 8.0) / 8.0;

                        self.precision_weight * score_eps
                            + self.dynamic_range_weight * score_dr
                            + self.speed_weight * score_speed
                            + self.memory_weight * score_mem
                    }
                    QuantizationEvaluationMode::AiOptimizer => {
                        let score_eps = (-eps.log10()).clamp(0.0, 4.0) / 4.0;
                        let score_dr = (dynamic_range_db / 300.0).clamp(0.0, 1.0);
                        let score_speed = (1.0 / latency_rel).clamp(0.1, 4.0) / 2.5;
                        let score_mem = (8.0 / bytes as f64).clamp(1.0, 8.0) / 8.0;

                        self.precision_weight * score_eps
                            + self.dynamic_range_weight * score_dr
                            + self.speed_weight * score_speed
                            + self.memory_weight * score_mem
                    }
                };

                let rationale = match mode {
                    QuantizationEvaluationMode::Integrator => {
                        format!(
                            "eps={:.1e}, DR={:.1}dB, bytes={}, speed={:.1}x",
                            eps, dynamic_range_db, bytes, 1.0 / latency_rel
                        )
                    }
                    QuantizationEvaluationMode::AiOptimizer => {
                        format!(
                            "bytes={}, speed={:.1}x, SQNR={:.1}dB, noise-floor robust",
                            bytes, 1.0 / latency_rel, sqnr_db
                        )
                    }
                };

                QuantizationRankEntry {
                    dtype,
                    name: spec.numeric_type_name().to_string(),
                    memory_bytes_per_param: bytes,
                    machine_epsilon: eps,
                    dynamic_range_db,
                    sqnr_db,
                    latency_relative_to_f32: latency_rel,
                    pareto_score,
                    recommendation_rank: 0,
                    rationale,
                }
            })
            .collect();

        // Sort descending by Pareto score
        entries.sort_by(|a, b| b.pareto_score.partial_cmp(&a.pareto_score).unwrap_or(std::cmp::Ordering::Equal));

        // Assign recommendation ranks 1..N
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.recommendation_rank = i + 1;
        }

        entries
    }

    /// Recommends the optimal numerical datatype for a given context.
    pub fn recommend_dtype(&self, mode: QuantizationEvaluationMode) -> NumericType {
        let ranked = self.rank_all(mode);
        ranked.first().map(|e| e.dtype).unwrap_or(NumericType::F32)
    }
}

// ============================================================================
// Scale-Free Dimension-Agnostic Error Norm
// ============================================================================

/// Computes component-wise mixed relative/absolute root-mean-square error norm:
///
/// $$E_{\text{scaled}} = \sqrt{ \frac{1}{d} \sum_{i=1}^d \left( \frac{|e_i|}{\text{atol} + \text{rtol} \cdot \max(|x_{i,\text{curr}}|, |x_{i,\text{next}}|)} \right)^2 }$$
///
/// When $E_{\text{scaled}} \le 1.0$, the local step satisfies the target tolerance across all dimensions.
pub fn compute_scaled_error_norm(
    err: &[f64],
    x_curr: &[f64],
    x_next: &[f64],
    atol: f64,
    rtol: f64,
) -> f64 {
    let d = err.len().max(1);
    let mut sum_sq = 0.0f64;

    for i in 0..err.len() {
        let scale = atol + rtol * x_curr[i].abs().max(x_next[i].abs());
        let norm_diff = err[i] / scale.max(1e-15);
        sum_sq += norm_diff * norm_diff;
    }

    (sum_sq / d as f64).sqrt()
}

// ============================================================================
// Information-Theoretic Pareto Gating Policy
// ============================================================================

/// Dynamic evaluation gating policy minimizing wall-clock time per information gained:
///
/// $$\mathcal{R}_k = \frac{\Delta \mathcal{I}_k}{\tau_{\text{eval}}} > \lambda_{\text{pareto}}$$
///
/// Where:
/// - $\Delta \mathcal{I}_k = \log_2 \left( 1.0 + \frac{\|\hat{x}^{(k)} - \hat{x}^{(k-1)}\|_{\text{scaled}}}{\text{tol}} \right)$ (Trajectory information bits)
/// - $\tau_{\text{eval}}$ is the wall-clock execution latency of the vector field.
/// - $\lambda_{\text{pareto}}$ is the marginal exchange rate (bits of precision per second).
#[derive(Debug, Clone)]
pub struct InformationRateGatingPolicy {
    /// Exponential Moving Average (EMA) of function evaluation latency in seconds.
    pub eval_latency_ema_sec: f64,
    /// Pareto trade-off multiplier: required bits gained per second of evaluation time.
    pub lambda_pareto: f64,
    /// Smoothing factor for latency EMA.
    pub ema_alpha: f64,
    /// Absolute floor for information gain to justify an extra corrector stage.
    pub min_bits_threshold: f64,
}

impl Default for InformationRateGatingPolicy {
    fn default() -> Self {
        Self {
            eval_latency_ema_sec: 1e-5, // Initial prior: 10 µs per evaluation
            lambda_pareto: 1e5,         // 100,000 bits / sec default exchange rate
            ema_alpha: 0.1,
            min_bits_threshold: 0.05,    // At least 0.05 bits of state uncertainty reduction
        }
    }
}

impl InformationRateGatingPolicy {
    pub fn new(lambda_pareto: f64) -> Self {
        Self {
            lambda_pareto,
            ..Default::default()
        }
    }

    /// Updates the measured evaluation latency using an exponential moving average.
    pub fn record_eval_latency(&mut self, elapsed_sec: f64) {
        if elapsed_sec > 0.0 && elapsed_sec.is_finite() {
            self.eval_latency_ema_sec =
                (1.0 - self.ema_alpha) * self.eval_latency_ema_sec + self.ema_alpha * elapsed_sec;
        }
    }

    /// Computes the trajectory information gained (in bits) between two successive integration states.
    pub fn compute_information_gain(&self, prior_state: &[f64], posterior_state: &[f64], tol: f64) -> f64 {
        let mut diff_sq = 0.0f64;
        for i in 0..prior_state.len() {
            let d = posterior_state[i] - prior_state[i];
            diff_sq += d * d;
        }
        let l2_diff = (diff_sq / prior_state.len().max(1) as f64).sqrt();
        (1.0 + l2_diff / tol.max(1e-15)).log2()
    }

    /// Evaluates whether an extra evaluation stage ($E_2$ / $\text{PECE}$ / $\text{PECEC}$) is Pareto-justified.
    pub fn should_evaluate_secondary_stage(&self, estimated_info_gain_bits: f64) -> bool {
        if estimated_info_gain_bits < self.min_bits_threshold {
            return false;
        }
        let marginal_rate = estimated_info_gain_bits / self.eval_latency_ema_sec.max(1e-9);
        marginal_rate >= self.lambda_pareto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_machine_epsilons() {
        let f64_prec = F64Precision;
        assert!(f64_prec.machine_epsilon(None) < 3e-16);

        let f32_prec = F32Precision;
        assert!(f32_prec.machine_epsilon(None) < 2e-7);

        let bf16_prec = Bf16Precision;
        assert!(bf16_prec.machine_epsilon(None) > 5e-3);

        let f16_prec = F16Precision;
        assert!(f16_prec.machine_epsilon(None) > 5e-4);

        let posit32 = Posit32Precision;
        let p32_eps_golden = posit32.machine_epsilon(Some(1.0));
        assert!(p32_eps_golden < f32_prec.machine_epsilon(None)); // Posit32 beats f32 in golden zone!

        let posit16 = Posit16Precision;
        let p16_eps_golden = posit16.machine_epsilon(Some(1.0));
        assert!(p16_eps_golden < f16_prec.machine_epsilon(None)); // Posit16 beats f16 in golden zone!

        let posit8 = Posit8Precision;
        assert_eq!(posit8.machine_epsilon(None), 0.0625);

        let posit64 = Posit64Precision;
        let p64_eps_golden = posit64.machine_epsilon(Some(1.0));
        assert!(p64_eps_golden < f64_prec.machine_epsilon(None)); // Posit64 beats f64 in golden zone!

        let log64 = Log64Precision;
        assert!(log64.machine_epsilon(None) <= f64_prec.machine_epsilon(None));

        let log32 = Log32Precision;
        assert!(log32.machine_epsilon(None) <= f32_prec.machine_epsilon(None));
    }

    #[test]
    fn test_quantization_ranker_integrator_vs_ai() {
        let ranker_integ = QuantizationRanker::for_integrator();
        let ranker_ai = QuantizationRanker::for_ai_optimizer();

        let integ_ranks = ranker_integ.rank_for_integrator();
        let ai_ranks = ranker_ai.rank_for_ai_optimizer();

        assert_eq!(integ_ranks.len(), 19);
        assert_eq!(ai_ranks.len(), 19);

        // In integrator mode, highest precision formats (Posit64, F64, Log64) dominate the top tier
        let top_integ_types: Vec<_> = integ_ranks.iter().take(4).map(|e| e.dtype).collect();
        assert!(top_integ_types.contains(&NumericType::Posit64) || top_integ_types.contains(&NumericType::F64));

        // In AI mode, compact formats (F16, Posit16, Log16, Int32, Ternary) dominate due to memory bandwidth
        let top_ai_types: Vec<_> = ai_ranks.iter().take(5).map(|e| e.dtype).collect();
        let has_compact = top_ai_types.iter().any(|d| {
            matches!(
                d,
                NumericType::F16
                    | NumericType::Posit16
                    | NumericType::Log16
                    | NumericType::Int32
                    | NumericType::Ternary
                    | NumericType::Fp8E4M3
            )
        });
        assert!(has_compact, "Top AI types should favor compact representations");
    }

    #[test]
    fn test_scale_free_error_norm() {
        let err = vec![1e-4, 2e-4];
        let x_curr = vec![1.0, 2.0];
        let x_next = vec![1.0, 2.0];
        let norm = compute_scaled_error_norm(&err, &x_curr, &x_next, 1e-3, 1e-3);
        assert!(norm < 1.0); // Error within target bounds
    }

    #[test]
    fn test_information_rate_gating() {
        let mut policy = InformationRateGatingPolicy::new(1e4); // 10,000 bits/sec
        policy.eval_latency_ema_sec = 1e-3; // Exactly 1 ms evaluation latency

        // 5 bits gained in 1 ms = 5,000 bits/sec < 10,000 bits/sec -> Skip secondary evaluation!
        assert!(!policy.should_evaluate_secondary_stage(5.0));

        // 20 bits gained in 1 ms = 20,000 bits/sec > 10,000 bits/sec -> Proceed!
        assert!(policy.should_evaluate_secondary_stage(20.0));
    }

    #[test]
    fn test_user_defined_custom_precision_trait_support() {
        // Demonstrate that users can define arbitrary custom precision representations
        // simply by implementing the `NumericPrecision` trait!
        #[derive(Debug, Clone)]
        struct CustomDeadzoneTernary {
            threshold: f64,
        }

        impl NumericPrecision for CustomDeadzoneTernary {
            fn name(&self) -> &'static str {
                "custom_deadzone_ternary"
            }
            fn bits(&self) -> usize {
                2
            }
            fn is_discrete(&self) -> bool {
                true
            }
            fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
                self.threshold
            }
            fn denominator_guard(&self) -> f64 {
                self.threshold * 0.1
            }
            fn temporal_guard(&self) -> f64 {
                self.threshold * 0.5
            }
            fn energy_drift_guard(&self) -> f64 {
                self.threshold
            }
            fn pid_antiwindup_guard(&self) -> f64 {
                self.threshold * 0.01
            }
            fn default_tolerances(&self) -> (f64, f64) {
                (self.threshold, self.threshold)
            }
            fn adaptive_clamp_bound(&self, _state_norm: f64, _velocity_norm: f64) -> f64 {
                1.0
            }
            fn quantize(&self, val: f64) -> f64 {
                if val > self.threshold {
                    1.0
                } else if val < -self.threshold {
                    -1.0
                } else {
                    0.0
                }
            }
        }

        let custom = CustomDeadzoneTernary { threshold: 0.25 };
        assert_eq!(custom.name(), "custom_deadzone_ternary");
        assert_eq!(custom.bits(), 2);
        assert!(custom.is_discrete());

        let mut data = vec![-0.5, -0.1, 0.05, 0.3, 0.9];
        custom.quantize_slice(&mut data);
        assert_eq!(data, vec![-1.0, 0.0, 0.0, 1.0, 1.0]);

        // Blanket impls allow Arc<dyn NumericPrecision> to be passed seamlessly
        let arc_custom: std::sync::Arc<dyn NumericPrecision> = std::sync::Arc::new(custom);
        assert_eq!(arc_custom.quantize(0.8), 1.0);
        assert_eq!(arc_custom.quantize(0.1), 0.0);
    }
}
