use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t02_support_vector_machines_and_kernel_methods as svm;
use std::path::PathBuf;

#[test]
fn test_02_support_vector_machines_and_kernel_methods() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/02_support_vector_machines_and_kernel_methods.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let y_cont = tensors.get("y_cont").expect("Missing y_cont");
    let golden_c_ridge = tensors.get("c_ridge").expect("Missing c_ridge");
    let golden_k_rbf = tensors.get("K_rbf").expect("Missing K_rbf");
    let golden_k_poly = tensors.get("K_poly").expect("Missing K_poly");
    let golden_k_nystrom = tensors.get("K_nystrom").expect("Missing K_nystrom");
    let landmarks = tensors.get("X_landmarks").expect("Missing X_landmarks");

    let gamma = 0.125f64;

    // 2. Parity Check: RBF Kernel Gram Matrix
    let k_rbf = svm::rbf_kernel(x, x, gamma)?;
    let diff_rbf = (&k_rbf - golden_k_rbf)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rbf < 1e-5,
        "Candle vs Golden RBF kernel error {diff_rbf:.2e} exceeds 1e-5"
    );

    // 3. Parity Check: Polynomial Kernel Gram Matrix
    let k_poly = svm::poly_kernel(x, x, 2.0, 1.0)?;
    let diff_poly = (&k_poly - golden_k_poly)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_poly < 1e-5,
        "Candle vs Golden Poly kernel error {diff_poly:.2e} exceeds 1e-5"
    );

    // 4. Parity Check: Kernel Ridge Regression Dual Coefficients
    let c_ridge = svm::solve_kernel_ridge(x, y_cont, gamma, 0.1)?;
    let diff_ridge = (&c_ridge - golden_c_ridge)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ridge < 1e-4,
        "Candle vs Golden Kernel Ridge dual coef error {diff_ridge:.2e} exceeds 1e-4"
    );

    // 5. Parity Check: Nyström Low-Rank Gram Approximation
    let k_nystrom = svm::nystrom_approximate_gram(x, landmarks, gamma, 1e-4)?;
    let diff_nystrom = (&k_nystrom - golden_k_nystrom)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nystrom < 1e-4,
        "Candle vs Golden Nystrom Gram error {diff_nystrom:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 02: all SVM and Kernel Method tests passed with machine precision!");
    Ok(())
}
