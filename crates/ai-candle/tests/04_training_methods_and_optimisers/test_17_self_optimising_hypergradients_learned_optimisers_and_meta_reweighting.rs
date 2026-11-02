use ai_candle::c04_training_methods_and_optimisers::t17_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting::HypergradientsUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_17_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting.safetensors");

    let module = HypergradientsUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 17 Verified Errors: {:?}", errs);
    assert!(*errs.get("hd_err").unwrap() < 1e-5);
    assert!(*errs.get("reweight_err").unwrap() < 1e-5);
    assert!(*errs.get("learned_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
