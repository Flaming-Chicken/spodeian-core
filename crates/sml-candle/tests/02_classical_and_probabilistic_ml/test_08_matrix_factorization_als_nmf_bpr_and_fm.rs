use candle_core::{Device, Result};
use sml_candle::c02_classical_and_probabilistic_ml::t08_matrix_factorization_als_nmf_bpr_and_fm as mf;
use std::path::PathBuf;

#[test]
fn test_08_matrix_factorization_als_nmf_bpr_and_fm() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/08_matrix_factorization_als_nmf_bpr_and_fm.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let w0 = tensors.get("w0").expect("Missing w0");
    let w_fm = tensors.get("w_fm").expect("Missing w_fm");
    let v_fm = tensors.get("V_fm").expect("Missing V_fm");
    let golden_fm_out = tensors.get("fm_out").expect("Missing fm_out");

    let p_als = tensors.get("P_als").expect("Missing P_als");
    let q_als = tensors.get("Q_als").expect("Missing Q_als");
    let golden_r_als = tensors.get("R_als_pred").expect("Missing R_als_pred");

    let w_updated = tensors.get("W_updated").expect("Missing W_updated");
    let h_updated = tensors.get("H_updated").expect("Missing H_updated");
    let golden_v_nmf = tensors.get("V_nmf_pred").expect("Missing V_nmf_pred");

    // 2. Parity Check: Factorization Machine Forward Pass
    let fm_out = mf::factorization_machine_forward(x, w0, w_fm, v_fm)?;
    let diff_fm = (&fm_out - golden_fm_out)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fm < 1e-4,
        "Candle vs Golden FM output error {diff_fm:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: ALS Low-Rank Reconstruction
    let r_als = mf::matrix_factorization_predict(p_als, q_als)?;
    let diff_als = (&r_als - golden_r_als)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_als < 1e-4,
        "Candle vs Golden ALS prediction error {diff_als:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: NMF Low-Rank Reconstruction
    let v_nmf = mf::matrix_factorization_predict(w_updated, &h_updated.t()?)?;
    let diff_nmf = (&v_nmf - golden_v_nmf)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nmf < 1e-4,
        "Candle vs Golden NMF prediction error {diff_nmf:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 08: FM, ALS & NMF tests passed with machine precision!");
    Ok(())
}
