use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t05_dimensionality_reduction_pca_lda_ica_tsne_umap as dr;
use std::path::PathBuf;

#[test]
fn test_05_dimensionality_reduction_pca_lda_ica_tsne_umap() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/05_dimensionality_reduction_pca_lda_ica_tsne_umap.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let y_class = tensors.get("y_class").expect("Missing y_class");
    let v_top = tensors.get("V_top").expect("Missing V_top");
    let golden_pca_proj = tensors.get("pca_proj").expect("Missing pca_proj");
    let golden_w_lda = tensors.get("w_lda").expect("Missing w_lda");
    let golden_lda_proj = tensors.get("lda_proj").expect("Missing lda_proj");

    // 2. Parity Check: Centered PCA Projection
    let (x_centered, _) = dr::center_matrix(x)?;
    let pca_proj = dr::pca_project(&x_centered, v_top)?;
    let diff_pca = (&pca_proj - golden_pca_proj)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pca < 1e-4,
        "Candle vs Golden PCA projection error {diff_pca:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: Fisher LDA Direction and Projection
    let w_lda = dr::solve_lda_direction(&x_centered, y_class, 1e-3)?;
    let diff_w_lda = (&w_lda - golden_w_lda)?.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff_w_lda < 1e-4,
        "Candle vs Golden LDA direction error {diff_w_lda:.2e} exceeds 1e-4"
    );

    let lda_proj = x_centered.matmul(&w_lda.unsqueeze(1)?)?.squeeze(1)?;
    let diff_lda = (&lda_proj - golden_lda_proj)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_lda < 1e-4,
        "Candle vs Golden LDA projection error {diff_lda:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 05: all Dimensionality Reduction tests passed with machine precision!");
    Ok(())
}
