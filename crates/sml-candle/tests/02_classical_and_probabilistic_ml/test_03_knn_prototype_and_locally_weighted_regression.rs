use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t03_knn_prototype_and_locally_weighted_regression as knn;
use std::path::PathBuf;

#[test]
fn test_03_knn_prototype_and_locally_weighted_regression() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/03_knn_prototype_and_locally_weighted_regression.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x_train = tensors.get("X_train").expect("Missing X_train");
    let y_train_cont = tensors.get("y_train_cont").expect("Missing y_train_cont");
    let x_test = tensors.get("X_test").expect("Missing X_test");
    let golden_dist_sq = tensors.get("dist_sq").expect("Missing dist_sq");
    let golden_knn_reg_pred = tensors.get("knn_reg_pred").expect("Missing knn_reg_pred");
    let golden_nw_pred = tensors.get("nw_pred").expect("Missing nw_pred");

    // 2. Parity Check: Pairwise Squared Distance
    let dist_sq = knn::pairwise_sq_dist(x_train, x_test)?;
    let diff_dist = (&dist_sq - golden_dist_sq)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_dist < 1e-4,
        "Candle vs Golden Pairwise dist_sq error {diff_dist:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: Distance-Weighted k-NN Regression Prediction
    let pred_knn = knn::knn_regress(x_train, y_train_cont, x_test, 5, 1e-5)?;
    let diff_knn = (&pred_knn - golden_knn_reg_pred)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_knn < 1e-4,
        "Candle vs Golden k-NN regression error {diff_knn:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: Nadaraya-Watson Kernel Regression Prediction
    let pred_nw = knn::nadaraya_watson_regress(x_train, y_train_cont, x_test, 1.0)?;
    let diff_nw = (&pred_nw - golden_nw_pred)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nw < 1e-4,
        "Candle vs Golden Nadaraya-Watson regression error {diff_nw:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 03: all k-NN and Local Regression tests passed with machine precision!");
    Ok(())
}
