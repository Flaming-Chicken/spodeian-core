use ai_candle::c04_training_methods_and_optimisers::t19_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian::NonBackpropUnifiedModule;
use candle_core::Result;
use std::path::PathBuf;

#[test]
fn test_19_candle_to_golden_parity() -> Result<()> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop/05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian.safetensors");

    let module = NonBackpropUnifiedModule::new(&weights_path)?;
    let errs = module.verify_invariants(1e-5)?;

    println!("Candle Topic 19 Verified Errors: {:?}", errs);
    assert!(*errs.get("ff_err").unwrap() < 1e-5);
    assert!(*errs.get("pc_err").unwrap() < 1e-5);
    assert!(*errs.get("dfa_err").unwrap() < 1e-5);
    assert!(*errs.get("oja_err").unwrap() < 1e-5);
    assert!(*errs.get("anchor_err").unwrap() < 1e-5);

    Ok(())
}
