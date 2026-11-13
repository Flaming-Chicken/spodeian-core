use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t04_clustering_kmeans_spectral_and_gmm_em as cluster;
use std::path::PathBuf;

#[test]
fn test_04_clustering_kmeans_spectral_and_gmm_em() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/04_clustering_kmeans_spectral_and_gmm_em.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let centers = tensors.get("centers").expect("Missing centers");
    let golden_kmeans_assign = tensors
        .get("kmeans_assignments")
        .expect("Missing kmeans_assignments");
    let golden_resp = tensors.get("resp").expect("Missing resp");
    let golden_l_sym = tensors.get("L_sym").expect("Missing L_sym");

    // 2. Parity Check: k-Means Cluster Assignments
    let assigns = cluster::kmeans_assign(x, centers)?;
    let diff_assigns = (&assigns - golden_kmeans_assign)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_assigns < 1e-4,
        "Candle vs Golden k-Means assignments mismatch: {diff_assigns}"
    );

    // 3. Parity Check: GMM E-step Responsibilities
    let (k, d) = centers.dims2()?;
    let pi = candle_core::Tensor::full(1.0f32 / (k as f32), (k,), &device)?;
    let var = candle_core::Tensor::ones((k, d), candle_core::DType::F32, &device)?;
    let resp = cluster::gmm_e_step(x, centers, &var, &pi)?;
    let diff_resp = (&resp - golden_resp)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_resp < 1e-4,
        "Candle vs Golden GMM E-step error {diff_resp:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: Symmetric Normalized Laplacian
    let l_sym = cluster::compute_symmetric_laplacian(x, 0.1)?;
    let diff_l = (&l_sym - golden_l_sym)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_l < 1e-4,
        "Candle vs Golden Laplacian error {diff_l:.2e} exceeds 1e-4"
    );

    println!(
        "Candle Topic 04: all Clustering, k-Means, and GMM tests passed with machine precision!"
    );
    Ok(())
}
