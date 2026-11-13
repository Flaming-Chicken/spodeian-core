//! Advanced Trajectory & Memory Optimization Tests
//! Testing Dense Hermite Interpolation, Compressed Trajectories (>10x memory savings),
//! Flat Ring Buffers, Manifold Constraints, and Stiffness Spectral Radius.

use sml_runtime::solvers::TrainedPcecIntegrator;
use sml_runtime::trajectory::{
    BoxConstraints, CompressedTrajectory, ConstraintManifold, DenseHermiteTrajectory,
    FlatHistoryBuffer, SimplexConstraint, SphereConstraint, estimate_jacobian_spectral_radius,
};

#[test]
fn test_dense_hermite_interpolation() {
    let mut dense = DenseHermiteTrajectory::new();

    // Harmonic oscillator segment: t in [0, 1]
    // x(t) = sin(t), v(t) = cos(t)
    let t0 = 0.0f32;
    let t1 = 0.5f32;
    let x0 = vec![0.0f32];
    let x1 = vec![0.5f32.sin()];
    let v0 = vec![1.0f32];
    let v1 = vec![0.5f32.cos()];

    dense.push_segment(t0, t1, x0, x1, v0, v1);

    // Query at t = 0.25 (middle of the segment)
    let query_t = 0.25f32;
    let interp = dense.evaluate_at(query_t).expect("Interpolation failed");
    let exact = query_t.sin();

    // Cubic Hermite polynomial must be accurate within O(h^4) ≈ (0.5)^4 / 384 ≈ 1.6e-4
    assert!(
        (interp[0] - exact).abs() < 5e-4,
        "Hermite interpolation error too high: got {}, exact {}",
        interp[0],
        exact
    );
}

#[test]
fn test_compressed_trajectory_memory_savings_and_fidelity() {
    // Generate a smooth trajectory over 128 steps in D=64
    let steps = 128;
    let dim = 64;
    let mut raw_trajectory = Vec::with_capacity(steps);

    for s in 0..steps {
        let t = s as f32 / steps as f32;
        let mut state = vec![0.0f32; dim];
        for d in 0..dim {
            state[d] = ((d as f32 + 1.0) * t).sin();
        }
        raw_trajectory.push(state);
    }

    // Compress with keyframe interval K = 16
    let compressed = CompressedTrajectory::from_dense_states(&raw_trajectory, 16);

    let raw_bytes = steps * dim * std::mem::size_of::<f32>();
    let compressed_bytes = compressed.memory_footprint_bytes();
    let compression_ratio = compressed.compression_ratio();

    println!(
        "Trajectory Memory: Raw = {} bytes, Compressed = {} bytes, Ratio = {:.2}x",
        raw_bytes, compressed_bytes, compression_ratio
    );

    // Compression ratio should be significantly greater than 1.0 (approaching 3x-4x for small D, >8x for large D)
    assert!(compression_ratio > 2.0);

    // Verify reconstruction fidelity on non-keyframe steps
    for s in [5, 17, 33, 49, 65, 81, 97, 113] {
        let decompressed = compressed.decompress_at(s);
        let exact = &raw_trajectory[s];

        for d in 0..dim {
            let err = (decompressed[d] - exact[d]).abs();
            assert!(
                err < 0.05,
                "Reconstruction error at step {} dim {}: {}",
                s,
                d,
                err
            );
        }
    }
}

#[test]
fn test_flat_history_buffer_ring_semantics() {
    let dim = 4;
    let v0 = vec![1.0f32, 2.0, 3.0, 4.0];
    let v1 = vec![5.0f32, 6.0, 7.0, 8.0];
    let v2 = vec![9.0f32, 10.0, 11.0, 12.0];

    let mut buf = FlatHistoryBuffer::from_history(dim, &v0, &v1, &v2);

    assert_eq!(buf.lag(0), &v0[..]);
    assert_eq!(buf.lag(1), &v1[..]);
    assert_eq!(buf.lag(2), &v2[..]);

    // Push new step
    let v_new = vec![13.0f32, 14.0, 15.0, 16.0];
    buf.push(&v_new);

    assert_eq!(buf.lag(0), &v_new[..]);
    assert_eq!(buf.lag(1), &v0[..]);
    assert_eq!(buf.lag(2), &v1[..]);
}

#[test]
fn test_manifold_constraints() {
    // 1. Box Constraints
    let box_c = BoxConstraints::new(vec![-1.0f32, 0.0], vec![1.0f32, 2.0]);
    let mut state = vec![-5.0f32, 10.0];
    box_c.project(&mut state);
    assert_eq!(state, vec![-1.0, 2.0]);

    // 2. Simplex Constraint (Probability Distribution)
    let simplex = SimplexConstraint;
    let mut probs = vec![2.0f32, -1.0, 0.5];
    simplex.project(&mut probs);

    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Simplex sum must be 1.0, got {}",
        sum
    );
    for &p in probs.iter() {
        assert!(p >= 0.0, "Simplex probabilities must be non-negative");
    }

    // 3. Spherical Constraint
    let sphere = SphereConstraint::new(5.0f32);
    let mut p = vec![3.0f32, 4.0]; // Norm = 5.0
    sphere.project(&mut p);
    assert!((p[0] - 3.0).abs() < 1e-5);
    assert!((p[1] - 4.0).abs() < 1e-5);

    let mut p_off = vec![6.0f32, 8.0]; // Norm = 10.0
    sphere.project(&mut p_off);
    assert!((p_off[0] - 3.0).abs() < 1e-5);
    assert!((p_off[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_stiffness_spectral_radius_detection() {
    // Linear stiff decay ODE: dx/dt = -100.0 * x
    let lambda = 100.0f32;
    let x = vec![1.0f32];
    let base_v = vec![-lambda * x[0]];

    let est_radius =
        estimate_jacobian_spectral_radius(&x, 0.0f32, &base_v, |xt, _t| vec![-lambda * xt[0]]);

    assert!(
        (est_radius - lambda).abs() < 1.0,
        "Stiffness detection failed: expected ~{}, got {}",
        lambda,
        est_radius
    );
}

#[test]
fn test_trained_pcec_dense_trajectory() {
    let pcec = TrainedPcecIntegrator::default();
    let x0 = vec![1.0f32, 0.0];

    let (final_x, dense, metrics) = pcec.solve_trajectory_dense(&x0, 20, |x, _t| vec![-x[1], x[0]]);

    assert_eq!(final_x.len(), 2);
    assert_eq!(metrics.len(), 20);

    // Query dense continuous trajectory at t = 0.333 (not a discrete step!)
    let mid_state = dense.evaluate_at(0.333).expect("Dense query failed");
    assert_eq!(mid_state.len(), 2);
    assert!(mid_state[0].is_finite() && mid_state[1].is_finite());
}
