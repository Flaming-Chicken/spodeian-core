use spodeian_ml_utils::stability::{
    nearest_psd_matrix, safe_div, safe_div_f32, symmetrize_matrix, ConvergenceError,
    IterationLimiter, SanitizeFloats,
};

#[test]
fn test_sanitize_floats_f64() {
    let mut data = vec![1.0, f64::NAN, 2.5, f64::INFINITY, -3.0, f64::NEG_INFINITY];
    assert!(!data.is_finite_vector());

    let replaced = data.replace_non_finite(0.0);
    assert_eq!(replaced, 3);
    assert!(data.is_finite_vector());
    assert_eq!(data, vec![1.0, 0.0, 2.5, 0.0, -3.0, 0.0]);

    data.clamp_range(-1.0, 2.0);
    assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0, -1.0, 0.0]);
}

#[test]
fn test_sanitize_floats_f32() {
    let mut data = vec![1.0f32, f32::NAN, 2.5, f32::INFINITY, -3.0, f32::NEG_INFINITY];
    assert!(!data.is_finite_vector());

    let replaced = data.replace_non_finite(0.0);
    assert_eq!(replaced, 3);
    assert!(data.is_finite_vector());
    assert_eq!(data, vec![1.0f32, 0.0, 2.5, 0.0, -3.0, 0.0]);

    data.clamp_range(-1.0, 2.0);
    assert_eq!(data, vec![1.0f32, 0.0, 2.0, 0.0, -1.0, 0.0]);
}

#[test]
fn test_safe_div() {
    assert_eq!(safe_div(10.0, 2.0, 0.0), 5.0);
    assert_eq!(safe_div(10.0, 0.0, -1.0), -1.0);
    assert_eq!(safe_div(10.0, 1e-16, -1.0), -1.0);
    assert_eq!(safe_div(f64::NAN, 2.0, -1.0), -1.0);
    assert_eq!(safe_div(10.0, f64::INFINITY, -1.0), -1.0);

    assert_eq!(safe_div_f32(10.0f32, 2.0f32, 0.0f32), 5.0f32);
    assert_eq!(safe_div_f32(10.0f32, 0.0f32, -1.0f32), -1.0f32);
}

#[test]
fn test_symmetrize_matrix() {
    let mut matrix = vec![
        1.0, 2.0,
        4.0, 3.0,
    ];
    symmetrize_matrix(&mut matrix, 2).expect("Symmetrization failed");
    assert_eq!(matrix, vec![1.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_nearest_psd_matrix_already_psd() {
    // Identity matrix is already PSD
    let identity = vec![
        1.0, 0.0,
        0.0, 1.0,
    ];
    let psd = nearest_psd_matrix(&identity, 2, 1e-8).expect("Projection failed");
    for (a, b) in identity.iter().zip(psd.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_nearest_psd_matrix_negative_eigenvalues() {
    // Matrix with eigenvalues 2 and -1:
    // [0.5, 1.5]
    // [1.5, 0.5]
    // Trace = 1, Det = 0.25 - 2.25 = -2. Eigenvalues: (1 +/- sqrt(1 - 4(-2)))/2 = (1 +/- 3)/2 = 2 and -1.
    let non_psd = vec![
        0.5, 1.5,
        1.5, 0.5,
    ];

    let eps = 1e-6;
    let psd = nearest_psd_matrix(&non_psd, 2, eps).expect("Projection failed");

    // The result must be symmetric
    assert!((psd[1] - psd[2]).abs() < 1e-10);

    // Verify positive semi-definiteness: x^T * A * x >= 0 for test vectors
    let test_vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2],
    ];

    for [x, y] in test_vectors {
        let quad = x * (psd[0] * x + psd[1] * y) + y * (psd[2] * x + psd[3] * y);
        assert!(
            quad >= -1e-10,
            "Quadratic form must be non-negative: got {quad}"
        );
    }
}

#[test]
fn test_nearest_psd_matrix_3x3_ill_conditioned() {
    // 3x3 ill-conditioned matrix
    let matrix = vec![
        1.0,  0.9,  0.9,
        0.9,  1.0,  0.9,
        0.9,  0.9, -0.5, // Corrupted entry causing negative eigenvalue
    ];

    let psd = nearest_psd_matrix(&matrix, 3, 1e-8).expect("3x3 projection failed");
    assert_eq!(psd.len(), 9);

    // Check symmetry
    for i in 0..3 {
        for j in 0..3 {
            assert!((psd[i * 3 + j] - psd[j * 3 + i]).abs() < 1e-10);
        }
    }

    // Verify positive diagonal
    for i in 0..3 {
        assert!(psd[i * 3 + i] > 0.0);
    }
}

#[test]
fn test_iteration_limiter_convergence() {
    let mut limiter = IterationLimiter::new(10, 1e-4);

    assert_eq!(limiter.step(1.0), Ok(false));
    assert_eq!(limiter.step(0.1), Ok(false));
    assert_eq!(limiter.step(0.001), Ok(false));
    assert_eq!(limiter.step(1e-5), Ok(true));
    assert_eq!(limiter.current_iteration, 4);
}

#[test]
fn test_iteration_limiter_exceeded() {
    let mut limiter = IterationLimiter::new(3, 1e-4);

    assert_eq!(limiter.step(1.0), Ok(false));
    assert_eq!(limiter.step(0.5), Ok(false));
    let err = limiter.step(0.2).unwrap_err();

    match err {
        ConvergenceError::MaxIterationsExceeded { iterations, .. } => {
            assert_eq!(iterations, 3);
        }
        _ => panic!("Expected MaxIterationsExceeded, got {err:?}"),
    }
}

#[test]
fn test_iteration_limiter_divergence() {
    let mut limiter = IterationLimiter::new(10, 1e-4);

    assert_eq!(limiter.step(1.0), Ok(false));
    let err = limiter.step(f64::NAN).unwrap_err();

    match err {
        ConvergenceError::Divergence { iteration, .. } => {
            assert_eq!(iteration, 1);
        }
        _ => panic!("Expected Divergence error, got {err:?}"),
    }
}
