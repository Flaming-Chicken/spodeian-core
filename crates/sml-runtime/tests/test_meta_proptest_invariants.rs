//! Randomized property & invariant tests for `SelfOptimizingMeta`, `SelfOptimizingMetaIntegrator`,
//! `Trajectory::evaluate`, and `loss::cross_entropy` scalar-guard safety.

use rand::{Rng, SeedableRng, rngs::StdRng};
use sml_runtime::candle_core::{Device, Tensor};
use sml_runtime::loss::{Reduction, cross_entropy};
use sml_runtime::optim::{SelfOptimizingMeta, SelfOptimizingMetaConfig};
use sml_runtime::solvers::{HamiltonianEnergyFn, SelfOptimizingMetaIntegrator};
use sml_runtime::trajectory::DenseHermiteTrajectory;
use std::sync::Arc;

#[test]
fn prop_self_optimizing_meta_finiteness_and_descent() {
    let mut rng = StdRng::seed_from_u64(0x5EED_4401);

    for trial in 0..48 {
        let dim = rng.random_range(4..=64);
        let lr = rng.random_range(1e-3f64..=5e-2f64);
        let cfg = SelfOptimizingMetaConfig {
            lr,
            ..Default::default()
        };
        let mut opt = SelfOptimizingMeta::new(Vec::new(), cfg);

        let mut theta: Vec<f32> = (0..dim)
            .map(|_| rng.random_range(-5.0f32..=5.0f32))
            .collect();
        let diag: Vec<f32> = (0..dim)
            .map(|i| 1.0 + (i as f32) * 0.5)
            .collect();

        let initial_loss: f32 = theta
            .iter()
            .zip(diag.iter())
            .map(|(&x, &d)| 0.5 * d * x * x)
            .sum();

        for step in 0..80 {
            let mut grads: Vec<f32> = theta
                .iter()
                .zip(diag.iter())
                .map(|(&x, &d)| d * x)
                .collect();
            // Inject occasional synthetic Cauchy-like gradient spike on half the trials
            if trial % 2 == 1 && step == 15 {
                grads[0] *= 1e4;
            }
            opt.step_slice_in_place_f32(0, &grads, &mut theta)
                .expect("step_slice_in_place_f32 should succeed");

            assert!(
                theta.iter().all(|v| v.is_finite()),
                "Non-finite parameter encountered at trial={trial} step={step}"
            );
        }

        let final_loss: f32 = theta
            .iter()
            .zip(diag.iter())
            .map(|(&x, &d)| 0.5 * d * x * x)
            .sum();
        assert!(
            final_loss < initial_loss,
            "Expected descent on convex quadratic at trial={trial}: initial={initial_loss}, final={final_loss}"
        );
    }
}

#[test]
fn prop_meta_integrator_decay_and_symplectic_invariants() {
    let mut rng = StdRng::seed_from_u64(0x5EED_4402);

    // 1. Linear decay invariance: dx/dt = -k*x => x(t1) = x0 * exp(-k*(t1-t0))
    for trial in 0..24 {
        let k = rng.random_range(0.25f32..=3.0f32);
        let x0: Vec<f32> = (0..4).map(|_| rng.random_range(0.5f32..=4.0f32)).collect();
        let t1 = rng.random_range(0.5f32..=2.0f32);
        let solver = SelfOptimizingMetaIntegrator::new(1e-4);

        let (xf, _diag, nfe) = solver.solve_interval(&x0, 0.0, t1, |x, _t| {
            x.iter().map(|&v| -k * v).collect()
        });
        assert!(nfe > 0, "Expected positive NFE on trial {trial}");
        assert!(xf.iter().all(|v| v.is_finite()));

        for (&got, &init) in xf.iter().zip(x0.iter()) {
            let expected = init * (-k * t1).exp();
            let err = (got - expected).abs();
            assert!(
                err < 2e-2 * init.abs().max(1.0),
                "Decay error too large on trial {trial}: got={got}, expected={expected}, err={err}"
            );
        }
    }

    // 2. Symplectic harmonic oscillator energy conservation: H = 0.5*(q^2 + p^2)
    for trial in 0..16 {
        let q0 = rng.random_range(0.5f32..=2.5f32);
        let p0 = rng.random_range(-1.5f32..=1.5f32);
        let target_e = 0.5 * (q0 * q0 + p0 * p0);

        let mut solver = SelfOptimizingMetaIntegrator::new(1e-4);
        solver.energy_target = Some(target_e);
        solver.energy_fn = Some(HamiltonianEnergyFn(Arc::new(|state: &[f32]| {
            0.5 * (state[0] * state[0] + state[1] * state[1])
        })));

        let (xf, _diag, _nfe) = solver.solve_interval(&[q0, p0], 0.0, 4.0, |s, _t| {
            vec![s[1], -s[0]]
        });
        assert!(xf.iter().all(|v| v.is_finite()));
        let final_e = 0.5 * (xf[0] * xf[0] + xf[1] * xf[1]);
        let drift = (final_e - target_e).abs() / target_e.max(1e-6);
        assert!(
            drift < 5e-2,
            "Symplectic energy drift too high on trial {trial}: target={target_e}, final={final_e}, drift={drift}"
        );
    }
}

#[test]
fn test_empty_trajectory_and_scalar_cross_entropy_guards() {
    let empty_traj: DenseHermiteTrajectory<f64> = DenseHermiteTrajectory::new();
    assert!(empty_traj.evaluate_at(0.5).is_none());

    let scalar_logits = Tensor::new(1.5f32, &Device::Cpu).expect("scalar tensor");
    let targets = Tensor::new(&[0u32], &Device::Cpu).expect("target tensor");
    let err = cross_entropy(&scalar_logits, &targets, 0.0, Reduction::Mean);
    assert!(
        err.is_err(),
        "cross_entropy must return Err on 0-D scalar logits instead of panicking"
    );
}
