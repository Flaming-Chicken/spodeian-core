use ai_candle::c04_training_methods_and_optimisers::t21_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse::ScalingLawsUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_21_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse.safetensors");

    let module = ScalingLawsUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 21 Verified Errors: {:?}", errs);
    assert!(*errs.get("chinchilla_err").unwrap() < 1e-5);
    assert!(*errs.get("doremi_err").unwrap() < 1e-5);
    assert!(*errs.get("minhash_err").unwrap() < 1e-5);
    assert!(*errs.get("collapse_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
