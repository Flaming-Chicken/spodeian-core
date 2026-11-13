use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t07_differentiable_trees_node_soft_gbdt_and_gam_nam as trees;
use std::path::PathBuf;

#[test]
fn test_07_differentiable_trees_node_soft_gbdt_and_gam_nam() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let w_split = tensors.get("W_split").expect("Missing W_split");
    let b_split = tensors.get("b_split").expect("Missing b_split");
    let leaf_values = tensors.get("leaf_values").expect("Missing leaf_values");
    let golden_node_out = tensors.get("node_out").expect("Missing node_out");

    let nam_w1 = tensors.get("nam_w1").expect("Missing nam_w1");
    let nam_b1 = tensors.get("nam_b1").expect("Missing nam_b1");
    let nam_w2 = tensors.get("nam_w2").expect("Missing nam_w2");
    let nam_b2 = tensors.get("nam_b2").expect("Missing nam_b2");
    let nam_bias0 = tensors.get("nam_bias0").expect("Missing nam_bias0");
    let golden_nam_out = tensors.get("nam_out").expect("Missing nam_out");

    let y_noisy = tensors.get("y_noisy").expect("Missing y_noisy");
    let golden_iso_fit = tensors.get("iso_fit").expect("Missing iso_fit");

    // 2. Parity Check: NODE Forward Pass
    let node_out = trees::node_forward(x, w_split, b_split, leaf_values)?;
    let diff_node = (&node_out - golden_node_out)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_node < 1e-4,
        "Candle vs Golden NODE output error {diff_node:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: NAM Forward Pass
    let nam_out = trees::nam_forward(x, nam_w1, nam_b1, nam_w2, nam_b2, nam_bias0)?;
    let diff_nam = (&nam_out - golden_nam_out)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nam < 1e-4,
        "Candle vs Golden NAM output error {diff_nam:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: PAV Isotonic Regression
    let y_vec = y_noisy.flatten_all()?.to_vec1::<f32>()?;
    let iso_fit = trees::pav_isotonic_regression(&y_vec);
    let golden_iso_vec = golden_iso_fit.flatten_all()?.to_vec1::<f32>()?;
    for i in 0..y_vec.len() {
        let diff = (iso_fit[i] - golden_iso_vec[i]).abs();
        assert!(
            diff < 1e-5,
            "Candle vs Golden PAV Isotonic error {diff:.2e} at {i} exceeds 1e-5"
        );
    }

    println!("Candle Topic 07: NODE, NAM & PAV Isotonic tests passed with machine precision!");
    Ok(())
}
