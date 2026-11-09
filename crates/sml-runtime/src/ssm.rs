//! State-Space Models (SSM) & Recurrence Kernels (`sml-runtime::ssm`).
//!
//! Includes vectorized continuous-time and discretized state-space operators,
//! specifically the Mamba2 state-space recurrence scan:
//! $$s_t = A s_{t-1} + B u_t$$
//!
//! Optimized for auto-vectorization across AVX2/AVX-512 (x86_64), NEON (aarch64),
//! and SIMD128 (wasm32) using 100% safe, bounds-checked idiomatic iterator zips.

#[inline(always)]
pub fn step_recurrence_f32(latent_state: &mut [f32], a_diag: &[f32], b_diag: &[f32], u_t: &[f32]) {
    let len = latent_state.len();
    assert!(
        a_diag.len() >= len && b_diag.len() >= len && u_t.len() >= len,
        "Mamba2 recurrence buffer dimension mismatch"
    );

    // Idiomatic zip allows LLVM to prove bounds and emit hardware vector instructions
    // (AVX2/AVX-512 on x86_64, NEON on aarch64, and SIMD128 on wasm32) with zero unsafe code.
    for (((s, &a), &b), &u) in latent_state
        .iter_mut()
        .zip(a_diag.iter())
        .zip(b_diag.iter())
        .zip(u_t.iter())
    {
        *s = a * *s + b * u;
    }
}
