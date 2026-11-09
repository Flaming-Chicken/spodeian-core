//! Umbral Calculus & Discrete Lattice Optimization for Pure Integer Types.
//!
//! Provides mathematically exact discrete difference operators, falling-factorial polynomial
//! transforms, and rational sub-grid residue accumulation across integer formats (INT32, INT16, INT8, Ternary).
//!
//! Grounded in the Rota-Roman umbral isomorphism:
//! - Derivative $D \longleftrightarrow \frac{1}{h} \ln(I + h\Delta_h)$
//! - Monomials $x^n \longleftrightarrow$ Falling Factorials $(x)_{n, h}$
//! - Taylor Series $\longleftrightarrow$ Newton-Gregory Series
//! - Eliminates 32-bit floating-point master shadow weights via rational residue tracking.

use serde::{Deserialize, Serialize};

/// Discrete umbral arithmetic operations on integer lattices.
pub trait UmbralLatticeScalar: Copy + Eq + Ord + std::fmt::Debug + Send + Sync + 'static {
    type Difference: Copy + Ord + std::fmt::Debug;

    /// Discrete forward difference: \Delta f(x) = f(x + 1) - f(x)
    fn forward_diff(self, next: Self) -> Self::Difference;

    /// Falling factorial: (x)_n = x * (x - 1) * ... * (x - n + 1)
    fn falling_factorial(self, n: usize) -> i64;

    /// Umbral binomial coefficient: \binom{x}{k} = (x)_k / k!
    fn umbral_binomial(self, k: usize) -> i64;
}

impl UmbralLatticeScalar for i32 {
    type Difference = i32;

    #[inline(always)]
    fn forward_diff(self, next: Self) -> Self::Difference {
        next.wrapping_sub(self)
    }

    fn falling_factorial(self, n: usize) -> i64 {
        if n == 0 {
            return 1;
        }
        let mut acc = 1i64;
        let x = self as i64;
        for i in 0..(n as i64) {
            acc = acc.wrapping_mul(x - i);
        }
        acc
    }

    fn umbral_binomial(self, k: usize) -> i64 {
        if k == 0 {
            return 1;
        }
        let mut num = 1i64;
        let mut den = 1i64;
        let x = self as i64;
        for i in 0..(k as i64) {
            num = num.wrapping_mul(x - i);
            den = den.wrapping_mul(i + 1);
        }
        num / den
    }
}

impl UmbralLatticeScalar for i16 {
    type Difference = i16;

    #[inline(always)]
    fn forward_diff(self, next: Self) -> Self::Difference {
        next.wrapping_sub(self)
    }

    fn falling_factorial(self, n: usize) -> i64 {
        (self as i32).falling_factorial(n)
    }

    fn umbral_binomial(self, k: usize) -> i64 {
        (self as i32).umbral_binomial(k)
    }
}

impl UmbralLatticeScalar for i8 {
    type Difference = i8;

    #[inline(always)]
    fn forward_diff(self, next: Self) -> Self::Difference {
        next.wrapping_sub(self)
    }

    fn falling_factorial(self, n: usize) -> i64 {
        (self as i32).falling_factorial(n)
    }

    fn umbral_binomial(self, k: usize) -> i64 {
        (self as i32).umbral_binomial(k)
    }
}

/// Exact rational residue decomposition $x = h \cdot (N + P/Q)$.
/// Tracks sub-lattice accumulation without requiring floating-point master shadow weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RationalResidue {
    pub numerator: i32,
    pub denominator: i32,
}

impl RationalResidue {
    pub const fn new(denominator: i32) -> Self {
        Self {
            numerator: 0,
            denominator,
        }
    }

    /// Accumulates an increment $\Delta P$ and propagates integer carries.
    /// Returns the signed integer carry $\Delta N \in \mathbb{Z}$.
    #[inline]
    pub fn accumulate_and_carry(&mut self, delta_p: i32) -> i32 {
        self.numerator += delta_p;
        let carry = self.numerator / self.denominator;
        self.numerator %= self.denominator;
        carry
    }
}

/// Pure integer optimizer state maintaining exact lattice coordinates and sub-grid residues.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UmbralIntegerState {
    pub len: usize,
    /// Native discrete integer coordinates $N \in \mathbb{Z}$
    pub lattice_coords: Vec<i32>,
    /// Exact sub-grid residue numerators $P_i$
    pub residue_numerators: Vec<i32>,
    /// Common residue denominator $Q$ (e.g. 1024 or 4096 for fine sub-lattice resolution)
    pub residue_denominator: i32,
    /// Discrete gradient / difference history
    pub prev_diffs: Vec<i32>,
    /// Discrete sign-flip EMA counters (scaled integer $[0, 255]$)
    pub flip_ema: Vec<u8>,
    /// Discrete Fisher variance accumulator
    pub fisher_var: Vec<i64>,
}

impl UmbralIntegerState {
    pub fn new(len: usize, residue_denominator: i32) -> Self {
        Self {
            len,
            lattice_coords: vec![0; len],
            residue_numerators: vec![0; len],
            residue_denominator: residue_denominator.max(1),
            prev_diffs: vec![0; len],
            flip_ema: vec![0; len],
            fisher_var: vec![1; len],
        }
    }

    /// Total memory footprint in bytes.
    pub fn byte_size(&self) -> usize {
        self.lattice_coords.len() * 4
            + self.residue_numerators.len() * 4
            + 4 // residue_denominator
            + self.prev_diffs.len() * 4
            + self.flip_ema.len()
            + self.fisher_var.len() * 8
    }

    /// Effective bytes per integer coordinate.
    pub fn bytes_per_element(&self) -> f32 {
        let n = self.len.max(1);
        self.byte_size() as f32 / n as f32
    }

    /// Executes an in-place discrete umbral step directly updating an external slice of integer coordinates.
    ///
    /// Synchronizes `coords` with the internal lattice state, avoids intermediate heap allocations,
    /// and ensures exact conservation of integer coordinates and rational residues.
    pub fn step_discrete_in_place(
        &mut self,
        coords: &mut [i32],
        diffs: &[i32],
        lr_numerator: i32,
        lr_denominator: i32,
    ) {
        let n = self.len.min(diffs.len()).min(coords.len());
        let q = self.residue_denominator as i64;
        let lr_num = lr_numerator as i64;
        let lr_den = lr_denominator.max(1) as i64;

        for i in 0..n {
            let d = diffs[i] as i64;
            let prev_d = self.prev_diffs[i] as i64;

            // 1. Discrete sign-flip detection on the integer lattice
            let is_sign_flip = (d > 0 && prev_d < 0) || (d < 0 && prev_d > 0);
            if is_sign_flip {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204 + 51) / 256).min(255) as u8;
            } else {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204) / 256) as u8;
            }
            self.prev_diffs[i] = diffs[i];

            // 2. Discrete Fisher variance update: v = (1023 v + d^2) / 1024
            let d_sq = d * d;
            self.fisher_var[i] = (self.fisher_var[i] * 1023 + d_sq) / 1024;
            let v_approx = (self.fisher_var[i] as f64).sqrt().max(1.0) as i64;

            // 3. Preconditioned step displacement in rational sub-lattice units
            let flip_damping = 256i64 + (self.flip_ema[i] as i64 * 128) / 256;
            let step_scaled = (d * lr_num * q * 256) / (v_approx * lr_den * flip_damping);

            // 4. Rational residue accumulation and carry propagation
            let mut residue = self.residue_numerators[i] as i64 - step_scaled;
            let carry = residue / q;
            residue %= q;

            let new_coord = coords[i] + carry as i32;
            coords[i] = new_coord;
            if i < self.lattice_coords.len() {
                self.lattice_coords[i] = new_coord;
            }
            self.residue_numerators[i] = residue as i32;
        }
    }

    /// Executes an in-place discrete umbral step directly updating an external slice of i16 coordinates.
    pub fn step_discrete_i16_in_place(
        &mut self,
        coords: &mut [i16],
        diffs: &[i16],
        lr_numerator: i32,
        lr_denominator: i32,
    ) {
        let n = self.len.min(diffs.len()).min(coords.len());
        let q = self.residue_denominator as i64;
        let lr_num = lr_numerator as i64;
        let lr_den = lr_denominator.max(1) as i64;

        for i in 0..n {
            let d = diffs[i] as i64;
            let prev_d = self.prev_diffs[i] as i64;

            let is_sign_flip = (d > 0 && prev_d < 0) || (d < 0 && prev_d > 0);
            if is_sign_flip {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204 + 51) / 256).min(255) as u8;
            } else {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204) / 256) as u8;
            }
            self.prev_diffs[i] = diffs[i] as i32;

            let d_sq = d * d;
            self.fisher_var[i] = (self.fisher_var[i] * 1023 + d_sq) / 1024;
            let v_approx = (self.fisher_var[i] as f64).sqrt().max(1.0) as i64;

            let flip_damping = 256i64 + (self.flip_ema[i] as i64 * 128) / 256;
            let step_scaled = (d * lr_num * q * 256) / (v_approx * lr_den * flip_damping);

            let mut residue = self.residue_numerators[i] as i64 - step_scaled;
            let carry = residue / q;
            residue %= q;

            let new_coord = (coords[i] as i32)
                .saturating_add(carry as i32)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            coords[i] = new_coord;
            if i < self.lattice_coords.len() {
                self.lattice_coords[i] = new_coord as i32;
            }
            self.residue_numerators[i] = residue as i32;
        }
    }

    /// Executes an in-place discrete umbral step directly updating an external slice of i8 coordinates.
    pub fn step_discrete_i8_in_place(
        &mut self,
        coords: &mut [i8],
        diffs: &[i8],
        lr_numerator: i32,
        lr_denominator: i32,
    ) {
        let n = self.len.min(diffs.len()).min(coords.len());
        let q = self.residue_denominator as i64;
        let lr_num = lr_numerator as i64;
        let lr_den = lr_denominator.max(1) as i64;

        for i in 0..n {
            let d = diffs[i] as i64;
            let prev_d = self.prev_diffs[i] as i64;

            let is_sign_flip = (d > 0 && prev_d < 0) || (d < 0 && prev_d > 0);
            if is_sign_flip {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204 + 51) / 256).min(255) as u8;
            } else {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204) / 256) as u8;
            }
            self.prev_diffs[i] = diffs[i] as i32;

            let d_sq = d * d;
            self.fisher_var[i] = (self.fisher_var[i] * 1023 + d_sq) / 1024;
            let v_approx = (self.fisher_var[i] as f64).sqrt().max(1.0) as i64;

            let flip_damping = 256i64 + (self.flip_ema[i] as i64 * 128) / 256;
            let step_scaled = (d * lr_num * q * 256) / (v_approx * lr_den * flip_damping);

            let mut residue = self.residue_numerators[i] as i64 - step_scaled;
            let carry = residue / q;
            residue %= q;

            let new_coord = (coords[i] as i32)
                .saturating_add(carry as i32)
                .clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            coords[i] = new_coord;
            if i < self.lattice_coords.len() {
                self.lattice_coords[i] = new_coord as i32;
            }
            self.residue_numerators[i] = residue as i32;
        }
    }

    /// Executes an in-place discrete umbral step directly updating an external slice of i64 coordinates.
    pub fn step_discrete_i64_in_place(
        &mut self,
        coords: &mut [i64],
        diffs: &[i64],
        lr_numerator: i64,
        lr_denominator: i64,
    ) {
        let n = self.len.min(diffs.len()).min(coords.len());
        let q = self.residue_denominator as i64;
        let lr_num = lr_numerator;
        let lr_den = lr_denominator.max(1);

        for i in 0..n {
            let d = diffs[i];
            let prev_d = self.prev_diffs[i] as i64;

            let is_sign_flip = (d > 0 && prev_d < 0) || (d < 0 && prev_d > 0);
            if is_sign_flip {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204 + 51) / 256).min(255) as u8;
            } else {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204) / 256) as u8;
            }
            self.prev_diffs[i] = (diffs[i].clamp(i32::MIN as i64, i32::MAX as i64)) as i32;

            let d_sq = (d.clamp(-1_000_000, 1_000_000)).pow(2);
            self.fisher_var[i] = (self.fisher_var[i] * 1023 + d_sq) / 1024;
            let v_approx = (self.fisher_var[i] as f64).sqrt().max(1.0) as i64;

            let flip_damping = 256i64 + (self.flip_ema[i] as i64 * 128) / 256;
            let step_scaled = (d * lr_num * q * 256) / (v_approx * lr_den * flip_damping);

            let mut residue = self.residue_numerators[i] as i64 - step_scaled;
            let carry = residue / q;
            residue %= q;

            let new_coord = coords[i].saturating_add(carry);
            coords[i] = new_coord;
            if i < self.lattice_coords.len() {
                self.lattice_coords[i] = (new_coord.clamp(i32::MIN as i64, i32::MAX as i64)) as i32;
            }
            self.residue_numerators[i] = residue as i32;
        }
    }

    /// Executes an in-place discrete umbral step directly updating ternary coordinates in `{-1, 0, +1}`.
    pub fn step_discrete_ternary_in_place(
        &mut self,
        coords: &mut [i8],
        diffs: &[i8],
        lr_numerator: i32,
        lr_denominator: i32,
    ) {
        let n = self.len.min(diffs.len()).min(coords.len());
        let q = self.residue_denominator as i64;
        let lr_num = lr_numerator as i64;
        let lr_den = lr_denominator.max(1) as i64;

        for i in 0..n {
            let d = diffs[i] as i64;
            let prev_d = self.prev_diffs[i] as i64;

            let is_sign_flip = (d > 0 && prev_d < 0) || (d < 0 && prev_d > 0);
            if is_sign_flip {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204 + 51) / 256).min(255) as u8;
            } else {
                self.flip_ema[i] = ((self.flip_ema[i] as u32 * 204) / 256) as u8;
            }
            self.prev_diffs[i] = diffs[i] as i32;

            let d_sq = d * d;
            self.fisher_var[i] = (self.fisher_var[i] * 1023 + d_sq) / 1024;
            let v_approx = (self.fisher_var[i] as f64).sqrt().max(1.0) as i64;

            let flip_damping = 256i64 + (self.flip_ema[i] as i64 * 128) / 256;
            let step_scaled = (d * lr_num * q * 256) / (v_approx * lr_den * flip_damping);

            let mut residue = self.residue_numerators[i] as i64 - step_scaled;
            let carry = residue / q;
            residue %= q;

            // Clamped to ternary set {-1, 0, +1}
            let new_coord = (coords[i] as i32 + carry as i32).clamp(-1, 1) as i8;
            coords[i] = new_coord;
            if i < self.lattice_coords.len() {
                self.lattice_coords[i] = new_coord as i32;
            }
            self.residue_numerators[i] = residue as i32;
        }
    }

    /// Executes an exact discrete umbral step on the internal integer lattice.
    ///
    /// Uses umbral rational interpolation:
    /// - Evaluates discrete differences $\Delta f_i$.
    /// - Tracks discrete sign-flips across lattice cells.
    /// - Scales displacement by $Q$ and accumulates into rational residues.
    /// - Updates lattice coordinates $N_i$ strictly via integer carries.
    pub fn step_discrete(
        &mut self,
        diffs: &[i32],
        lr_numerator: i32,
        lr_denominator: i32,
    ) {
        let mut coords = std::mem::take(&mut self.lattice_coords);
        self.step_discrete_in_place(&mut coords, diffs, lr_numerator, lr_denominator);
        self.lattice_coords = coords;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falling_factorial_basic_sequence() {
        // (x)_3 = x(x-1)(x-2)
        assert_eq!(5i32.falling_factorial(0), 1);
        assert_eq!(5i32.falling_factorial(1), 5);
        assert_eq!(5i32.falling_factorial(2), 20); // 5 * 4
        assert_eq!(5i32.falling_factorial(3), 60); // 5 * 4 * 3
        assert_eq!(5i32.falling_factorial(4), 120); // 5 * 4 * 3 * 2
        assert_eq!(5i32.falling_factorial(5), 120); // 5 * 4 * 3 * 2 * 1
        assert_eq!(5i32.falling_factorial(6), 0); // Factor (5-5) = 0
    }

    #[test]
    fn test_delta_operator_on_falling_factorials() {
        // Delta (x)_n = n * (x)_{n-1}
        let x = 6i32;
        let n = 3;
        let val_x = x.falling_factorial(n);
        let val_next = (x + 1).falling_factorial(n);
        let delta = val_next - val_x;
        let expected = (n as i64) * x.falling_factorial(n - 1);
        assert_eq!(delta, expected);
    }

    #[test]
    fn test_rational_residue_carry_propagation() {
        let mut res = RationalResidue::new(100);
        assert_eq!(res.accumulate_and_carry(40), 0);
        assert_eq!(res.numerator, 40);

        // Accumulate 70 -> total 110 -> carry 1, remainder 10
        assert_eq!(res.accumulate_and_carry(70), 1);
        assert_eq!(res.numerator, 10);

        // Negative accumulation
        assert_eq!(res.accumulate_and_carry(-25), 0);
        assert_eq!(res.numerator, -15);
    }

    #[test]
    fn test_umbral_integer_convergence_on_discrete_bowl() {
        let n = 64;
        let mut state = UmbralIntegerState::new(n, 1024);
        for i in 0..n {
            state.lattice_coords[i] = 100; // Start at discrete displacement +100
        }

        // Discrete bowl: f(x) = sum x_i^2 -> \Delta f(x)_i = 2 x_i + 1
        for _ in 0..150 {
            let diffs: Vec<i32> = state.lattice_coords.iter().map(|&x| 2 * x + 1).collect();
            state.step_discrete(&diffs, 1, 1);
        }

        // Must contract from 100 down toward 0 on the integer lattice
        let max_abs = state.lattice_coords.iter().map(|&x| x.abs()).max().unwrap();
        assert!(max_abs <= 2, "Umbral integer state must contract to discrete neighborhood of origin, got max {}", max_abs);
    }

    #[test]
    fn test_umbral_integer_in_place_equivalence() {
        let n = 32;
        let mut state_a = UmbralIntegerState::new(n, 2048);
        let mut state_b = UmbralIntegerState::new(n, 2048);
        let mut external_coords = vec![42i32; n];
        for i in 0..n {
            state_a.lattice_coords[i] = 42;
            state_b.lattice_coords[i] = 42;
        }

        let diffs: Vec<i32> = (0..n).map(|i| (i as i32 * 3) - 45).collect();

        state_a.step_discrete(&diffs, 5, 2);
        state_b.step_discrete_in_place(&mut external_coords, &diffs, 5, 2);

        assert_eq!(state_a.lattice_coords, external_coords);
        assert_eq!(state_b.lattice_coords, external_coords);
        assert_eq!(state_a.residue_numerators, state_b.residue_numerators);
        assert_eq!(state_a.prev_diffs, state_b.prev_diffs);
        assert_eq!(state_a.flip_ema, state_b.flip_ema);
        assert_eq!(state_a.fisher_var, state_b.fisher_var);
    }

    #[test]
    fn test_umbral_integer_i16_and_i8_in_place_convergence() {
        let n = 16;
        let mut state_i16 = UmbralIntegerState::new(n, 1024);
        let mut coords_i16 = vec![50i16; n];
        for _ in 0..100 {
            let diffs: Vec<i16> = coords_i16.iter().map(|&x| 2 * x + 1).collect();
            state_i16.step_discrete_i16_in_place(&mut coords_i16, &diffs, 1, 1);
        }
        let max_abs_i16 = coords_i16.iter().map(|&x| x.abs()).max().unwrap();
        assert!(max_abs_i16 <= 2, "i16 state must contract to origin, got max {}", max_abs_i16);

        let mut state_i8 = UmbralIntegerState::new(n, 1024);
        let mut coords_i8 = vec![30i8; n];
        for _ in 0..100 {
            let diffs: Vec<i8> = coords_i8.iter().map(|&x| 2 * x + 1).collect();
            state_i8.step_discrete_i8_in_place(&mut coords_i8, &diffs, 1, 1);
        }
        let max_abs_i8 = coords_i8.iter().map(|&x| x.abs()).max().unwrap();
        assert!(max_abs_i8 <= 2, "i8 state must contract to origin, got max {}", max_abs_i8);
    }

    #[test]
    fn test_umbral_integer_i64_and_ternary_in_place_convergence() {
        let n = 16;
        // Test i64 lattice convergence
        let mut state_i64 = UmbralIntegerState::new(n, 2048);
        let mut coords_i64 = vec![100i64; n];
        for _ in 0..250 {
            let diffs: Vec<i64> = coords_i64.iter().map(|&x| 2 * x + 1).collect();
            state_i64.step_discrete_i64_in_place(&mut coords_i64, &diffs, 1, 1);
        }
        let max_abs_i64 = coords_i64.iter().map(|&x| x.abs()).max().unwrap();
        assert!(max_abs_i64 <= 2, "i64 state must contract to origin, got max {}", max_abs_i64);

        // Test ternary {-1, 0, +1} lattice convergence
        let mut state_ternary = UmbralIntegerState::new(n, 1024);
        let mut coords_ternary = vec![1i8; n];
        for _ in 0..100 {
            let diffs: Vec<i8> = coords_ternary.iter().map(|&x| 2 * x + 1).collect();
            state_ternary.step_discrete_ternary_in_place(&mut coords_ternary, &diffs, 1, 1);
            // All coordinates must remain strictly in {-1, 0, 1}
            for &c in &coords_ternary {
                assert!((-1..=1).contains(&c), "Ternary coordinate {} out of range", c);
            }
        }
        let max_abs_ternary = coords_ternary.iter().map(|&x| x.abs()).max().unwrap();
        assert!(max_abs_ternary <= 1, "Ternary state must remain in bounds, got max {}", max_abs_ternary);
    }
}
