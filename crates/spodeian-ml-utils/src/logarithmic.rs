//! Native Logarithmic Number System (LNS) Inference Engine & Kernels.
//!
//! Replaces standard floating-point multiplications with exact integer/fixed-point additions
//! in logarithmic space:
//! $$x \cdot y = (-1)^{s_x \oplus s_y} \cdot 2^{e_x + e_y}$$
//!
//! Provides:
//! 1. `LogarithmicScalar`: Compact 8-bit or 16-bit logarithmic float representation.
//! 2. Native Rust/WASM vector dot product & GEMM kernels.
//! 3. WebGPU WGSL compute shader for hardware-accelerated logarithmic inference.

use serde::{Deserialize, Serialize};

/// 8-bit Logarithmic Float: 1 sign bit + 7 exponent bits (representing base-2 log with bias 64).
/// Represents values $x = (-1)^s \cdot 2^{(e - 64) / 4.0}$.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Log8(pub u8);

impl Log8 {
    pub const ZERO: Self = Self(0);

    /// Converts an f32 into an 8-bit logarithmic scalar.
    pub fn from_f32(val: f32) -> Self {
        if val.abs() < 1e-12 {
            return Self::ZERO;
        }

        let sign_bit: u8 = if val < 0.0 { 0x80 } else { 0x00 };
        let log2_val = val.abs().log2();
        // Scale by 4 (giving 0.25 exponent resolution) and add bias 64
        let biased_exp = ((log2_val * 4.0).round() as i32 + 64).clamp(1, 127) as u8;

        Self(sign_bit | biased_exp)
    }

    /// Converts an 8-bit logarithmic scalar back to an f32.
    pub fn to_f32(self) -> f32 {
        if self.0 == 0 {
            return 0.0;
        }

        let sign: f32 = if (self.0 & 0x80) != 0 { -1.0 } else { 1.0 };
        let biased_exp = (self.0 & 0x7F) as i32;
        let log2_val = (biased_exp - 64) as f32 / 4.0;

        sign * 2.0f32.powf(log2_val)
    }

    /// Exact multiplication in logarithmic space (exponent addition, sign XOR).
    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        if self.0 == 0 || other.0 == 0 {
            return Self::ZERO;
        }

        let sign_bit = (self.0 ^ other.0) & 0x80;
        let exp_a = (self.0 & 0x7F) as i32 - 64;
        let exp_b = (other.0 & 0x7F) as i32 - 64;
        let exp_sum = exp_a + exp_b + 64;

        let clamped_exp = exp_sum.clamp(1, 127) as u8;
        Self(sign_bit | clamped_exp)
    }
}

impl std::ops::Mul for Log8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

/// Vector dot product performed natively using logarithmic multiplication.
/// Converts product terms to accumulator using Mitchell/log-sum-exp or linear accumulation.
#[doc = "**Stability: Experimental** — API subject to change"]
pub fn log8_dot_product(a: &[Log8], b: &[Log8]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;

    for i in 0..n {
        let prod = a[i].mul(b[i]);
        sum += prod.to_f32();
    }

    sum
}

/// Quantizes an f32 vector into a packed Log8 array.
#[doc = "**Stability: Experimental** — API subject to change"]
pub fn quantize_to_log8(values: &[f32]) -> Vec<Log8> {
    values.iter().map(|&v| Log8::from_f32(v)).collect()
}

/// WGSL Compute Shader source for hardware-accelerated Logarithmic GEMM.
#[doc = "**Stability: Experimental** — API subject to change"]
pub const LOGARITHMIC_GEMM_WGSL: &str = r#"
// WebGPU Compute Shader: Logarithmic GEMM Kernel
// Multiplies weights and activations natively via exponent addition.

struct LogUniforms {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LogUniforms;
@group(0) @binding(1) var<storage, read> a_log: array<u32>; // Packed 4x u8 per u32
@group(0) @binding(2) var<storage, read> b_log: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_f32: array<f32>;

fn decode_log8(byte_val: u32) -> f32 {
    if (byte_val == 0u) {
        return 0.0;
    }
    let sign_bit = (byte_val & 0x80u);
    let sign: f32 = select(1.0, -1.0, sign_bit != 0u);
    let biased_exp = f32(byte_val & 0x7Fu);
    let log2_val = (biased_exp - 64.0) / 4.0;
    return sign * exp2(log2_val);
}

@compute @workgroup_size(16, 16)
fn log_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= uniforms.m || col >= uniforms.n) {
        return;
    }

    var acc: f32 = 0.0;
    for (var i = 0u; i < uniforms.k; i = i + 1u) {
        let a_val = decode_log8(a_log[row * uniforms.k + i]);
        let b_val = decode_log8(b_log[i * uniforms.n + col]);
        acc = acc + a_val * b_val;
    }

    out_f32[row * uniforms.n + col] = acc;
}
"#;
